from types import MappingProxyType
from typing import ClassVar, Literal, Optional
import jax.numpy as jnp
import jax.tree as jt 

import equinox as eqx
from jax_cookbook import is_type, is_module
import jax_cookbook.tree as jtree

from feedbax.intervene import schedule_intervenor
import feedbax.plotly as fbp

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.plot import WHERE_PLOT_PLANT_VARS
from rnns_learn_robust_motor_policies.plot import PLANT_VAR_LABELS
from rnns_learn_robust_motor_policies.types import Responses
from rnns_learn_robust_motor_policies.analysis.state_utils import vmap_eval_ensemble
from rnns_learn_robust_motor_policies.types import ImpulseAmpTuple, LDict
from rnns_learn_robust_motor_policies.perturbations import feedback_impulse


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
                1.0, #impulse_amplitude[pert_var_names[feedback_var_idx]],
                hps.pert.duration,
                feedback_var_idx,   
                hps.pert.start_step,
            ),
            default_active=False,
            stage_name="update_queue",
        ),
        LDict.of("pert__fb_var")(dict(pos=0, vel=1)),
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
    feedback_var_idxs = LDict.of("pert__fb_var")(dict(zip(PERT_VAR_NAMES, range(len(PERT_VAR_NAMES)))))
    coord_idxs = dict(zip(COORD_NAMES, range(len(COORD_NAMES))))
    
    impulse_xy_conditions = LDict.of("pert__fb_var").fromkeys(PERT_VAR_NAMES, dict.fromkeys(COORD_NAMES))
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


def setup_eval_tasks_and_models(task_base, models_base, hps):
    impulse_amplitudes = jt.map(
        lambda max_amp: jnp.linspace(0, max_amp, hps.pert.n_amplitudes + 1)[1:],
        hps.pert.amp_max,
    )
    hps.pert.amp = impulse_amplitudes

    # impulse_end_step = hps.pert.start_step + hps.pert.duration
    # TODO: Move extra info to another function? Or return it here.
    # impulse_time_idxs = slice(hps.pert.start_step, impulse_end_step)

    # For the example trajectories and aligned profiles, we'll only plot one of the impulse amplitudes. 

    # i_impulse_amp_plot = -1  # The largest amplitude perturbation
    # impulse_amplitude_plot = {
    #     pert_var: v[i_impulse_amp_plot] for pert_var, v in impulse_amplitudes.items()
    # }

    all_tasks, all_models, impulse_directions = SETUP_FUNCS_BY_DIRECTION[hps.pert.direction](
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


def eval_func(key_eval, hps, models, task):
    """Vmap over impulse amplitude."""
    return eqx.filter_vmap(
        lambda amplitude: vmap_eval_ensemble(
            key_eval,
            hps,
            models, 
            task_with_imp_amplitude(task, amplitude), 
        )
    )(hps.pert.amps)
    
    

class SingleImpulseAmplitude(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType({})
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    i_impulse_amp_plot: int = -1 
    
    def compute(self, models, tasks, states, hps, **dependencies):
        #! This was used in getting `pert_amp` for `add_evaluation_figure` params; I don't think we need it anymore
        # impulse_amplitude_plot = {
        #     pert_var: v[self.i_impulse_amp_plot] for pert_var, v in hps.pert.amp.items()
        # }
        
        return jt.map(
            lambda t: t[self.i_impulse_amp_plot],
            states,
            is_leaf=is_type(ImpulseAmpTuple),
        )    
    

class ExampleTrialSets(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        single_impulse_amp_states=SingleImpulseAmplitude,
    ))
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    i_trial: int = 0
    i_replicate: Optional[int] = None

    def compute(self, models, tasks, states, hps, *, single_impulse_amp_states, **dependencies):
        return jt.map(WHERE_PLOT_PLANT_VARS, states, is_leaf=is_module)
        
        # # Split up the impulse amplitudes from array dim 0, into a tuple part of the PyTree,
        # # and unzip them so `ExamplePlotVars` is on the inside
        # plot_states = jt.map(
        #     lambda plot_vars: jtree.unzip(
        #         jt.map(
        #             lambda arr: ImpulseAmpTuple(arr),
        #             plot_vars,
        #         ),
        #         ImpulseAmpTuple,
        #     ),
        #     plot_states,
        #     is_leaf=is_type(Responses),
        # )
        
        # # Only plot the strongest impulse amplitude, here
        # # (This makes the last step kind of superfluous but if we need to change this 
        # # again later, it might be convenient for the impulse amplitudes to be part of 
        # # the PyTree structure)
        # plot_states = jt.map(
        #     lambda t: t[i_impulse_amp_plot],
        #     plot_states,
        #     is_leaf=is_type(ImpulseAmpTuple),
        # )

    def make_figs(self, models, tasks, states, hps, *, result, replicate_info, **dependencies):
        
        if self.i_replicate is None:
            get_replicate = lambda train_std: replicate_info[train_std]['best_replicate']
        else:
            get_replicate = lambda _: self.i_replicate
            
        figs = jt.map(  
            lambda states: LDict.of("train__pert__std")({
                train_std: fbp.trajectories_2D(
                    jtree.take_multi(
                        plot_vars, 
                        [self.i_trial, get_replicate(train_std)],
                        [0, 1]
                    ),
                    var_labels=PLANT_VAR_LABELS,
                    axes_labels=('x', 'y'),
                    curves_mode='markers+lines',
                    ms=3,
                    scatter_kws=dict(line_width=0.75),
                    layout_kws=dict(
                        width=100 + len(PLANT_VAR_LABELS) * 300,
                        height=400,
                        legend_tracegroupgap=1,
                    ),
                )
                for train_std, plot_vars in states.items()
            }),
            result,
            is_leaf=LDict.is_of("train__pert__std"),
        )  
        return figs
    

class ResponseTrajectories(AbstractAnalysis):
    dependencies: ClassVar[MappingProxyType[str, type[AbstractAnalysis]]] = MappingProxyType(dict(
        single_impulse_amp_states=SingleImpulseAmplitude,
    ))
    variant: Optional[str] = "full"
    conditions: tuple[str, ...] = ()
    
    def make_figs(self, models, tasks, states, hps, *, result, **dependencies):
        figs = {}  # Define figs to fix the linter error
        return figs        


ALL_ANALYSES = []