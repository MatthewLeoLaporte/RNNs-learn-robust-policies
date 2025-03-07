from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, Literal, Optional
import jax.numpy as jnp
import jax.tree as jt 

import equinox as eqx
from jax_cookbook import is_type, is_module
import jax_cookbook.tree as jtree

from feedbax.intervene import schedule_intervenor
import feedbax.plotly as fbp

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisInputData
from rnns_learn_robust_motor_policies.analysis.effector import Effector_SingleEval
from rnns_learn_robust_motor_policies.plot import PLANT_VAR_LABELS, WHERE_PLOT_PLANT_VARS
from rnns_learn_robust_motor_policies.analysis.state_utils import vmap_eval_ensemble
from rnns_learn_robust_motor_policies.types import ImpulseAmpTuple, LDict, unflatten_dict_keys
from rnns_learn_robust_motor_policies.perturbations import feedback_impulse


ID = "1-2"


#! TODO: Move
PERT_VAR_NAMES = ('fb_pos', 'fb_vel')
COORD_NAMES = ('x', 'y')

COLOR_FUNCS = dict()


components_plot: Literal['xy', 'aligned'] = 'aligned'
components_labels = dict(
    xy=COORD_NAMES,
    aligned=(r'\parallel', r'\bot')
)
components_names = dict(
    xy=COORD_NAMES,
    aligned=('parallel', 'orthogonal'),
)


def _setup_rand(task_base, models_base, hps):
    """Impulses in random directions, i.e. uniform angles about the effector."""
    all_tasks, all_models = jtree.unzip(jt.map(
        lambda feedback_var_idx: schedule_intervenor(
            task_base, models_base,
            lambda model: model.step.feedback_channels[0],  # type: ignore
            feedback_impulse(  
                hps.model.n_steps,
                1.0,  # Will be varied later
                hps.pert.duration,
                feedback_var_idx,   
                hps.pert.start_step,
            ),
            default_active=False,
            stage_name="update_queue",
        ),
        LDict.of("pert__var")(dict(fb_pos=0, fb_vel=1)),
        is_leaf=is_type(tuple),
    ))

    # Get the perturbation directions, for later:
    #? I think these values are equivalent to `line_vec` in the functions in `state_utils`
    impulse_directions = jt.map(
        lambda task: task.validation_trials.intervene['ConstantInput'].arrays[:, hps.pert.start_step],
        all_tasks,
        is_leaf=is_module,
    )
    return all_tasks, all_models, impulse_directions


def _setup_xy(task_base, models_base, hps):
    """Impulses only in the x and y directions."""
    feedback_var_idxs = LDict.of("pert__var")(
        dict(zip(PERT_VAR_NAMES, range(len(PERT_VAR_NAMES))))
    )
    coord_idxs = dict(zip(COORD_NAMES, range(len(COORD_NAMES))))
    
    impulse_xy_conditions = LDict.of("pert__var").fromkeys(PERT_VAR_NAMES, dict.fromkeys(COORD_NAMES))
    impulse_xy_conditions_keys = jtree.key_tuples(
        impulse_xy_conditions, keys_to_strs=True, is_leaf=lambda x: x is None,
    )

    all_tasks, all_models = jtree.unzip(jt.map(
        lambda ks: schedule_intervenor(
            task_base, models_base,
            lambda model: model.step.feedback_channels[0],  # type: ignore
            feedback_impulse(
                hps.model.n_steps,
                1.0, 
                hps.pert.duration,
                feedback_var_idxs[ks[0]],  
                hps.pert.start_step,
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

I_IMPULSE_AMP_PLOT = -1  # The largest amplitude perturbation


def setup_eval_tasks_and_models(task_base, models_base, hps):

    impulse_end_step = hps.pert.start_step + hps.pert.duration
    impulse_time_idxs = slice(hps.pert.start_step, impulse_end_step)


    all_tasks, all_models, impulse_directions = SETUP_FUNCS_BY_DIRECTION[hps.pert.direction](
        task_base, models_base, hps
    )
    
    impulse_amplitudes = jt.map(
        lambda max_amp: jnp.linspace(0, max_amp, hps.pert.n_amps + 1)[1:],
        LDict.of("pert__var").from_ns(hps.pert.amp_max),
    )

    # For the example trajectories and aligned profiles, we'll only plot one of the impulse amplitudes. 
    impulse_amplitude_plot = {
        pert_var: v[I_IMPULSE_AMP_PLOT] for pert_var, v in impulse_amplitudes.items()
    }
    
    all_hps = jt.map(
        lambda amps: hps | unflatten_dict_keys(dict(pert__amps=amps)), 
        impulse_amplitudes,
    )
    
    extras = SimpleNamespace(
        impulse_directions=impulse_directions,
        impulse_time_idxs=impulse_time_idxs,
        impulse_amplitude_plot=impulse_amplitude_plot,
    )
    
    return all_tasks, all_models, all_hps, extras


def task_with_imp_amplitude(task, impulse_amplitude):
    """Returns a task with the given disturbance amplitude."""
    return eqx.tree_at(
        lambda task: task.intervention_specs.validation['ConstantInput'].intervenor.params.scale,
        task,
        impulse_amplitude,
    ) 


def eval_func(key_eval, hps, models, task):
    """Vmap over impulse amplitude."""
    states = eqx.filter_vmap(
        lambda amplitude: vmap_eval_ensemble(
            key_eval,
            hps,
            models, 
            task_with_imp_amplitude(task, amplitude), 
        ),
    )(hps.pert.amps)
    
    # I am not sure why this moveaxis is necessary. 
    # I tried using `out_axes=2` (with or without `in_axes=0`) and 
    # the result has the trial (axis 0) and replicate (axis 1) swapped.
    # (I had expected vmap to simply insert the new axis in the indicated position.)
    return jt.map(
        lambda arr: jnp.moveaxis(arr, 0, 2),
        states,
    )
    

class States_SingleImpulseAmplitude(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType({})
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    i_impulse_amp_plot: int = -1 
    
    def compute(self, data: AnalysisInputData, **dependencies):
        #! This was used in getting `pert_amp` for `add_evaluation_figure` params; I don't think we need it anymore
        # impulse_amplitude_plot = {
        #     pert_var: v[self.i_impulse_amp_plot] for pert_var, v in hps.pert.amp.items()
        # }
        
        return jt.map(
            lambda t: t[self.i_impulse_amp_plot],
            data.states,
            is_leaf=is_type(ImpulseAmpTuple),
        )    
    

class ResponseTrajectories(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        single_impulse_amp_states=States_SingleImpulseAmplitude,
    ))
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    
    def make_figs(self, data: AnalysisInputData, *, result, **dependencies):
        figs = {}  # Define figs to fix the linter error
        return figs        


VARIANT = "full"

ALL_ANALYSES = [
    Effector_SingleEval(
        variant=VARIANT,
        #! TODO: This doesn't result in the impulse amplitude *values* showing up in the legend!
        #! (could try to access `colorscale_key` from `hps`, in `Effector_SingleEval`)
        legend_title="Impulse amplitude",
        colorscale_key='pert__amp',
    ),
]