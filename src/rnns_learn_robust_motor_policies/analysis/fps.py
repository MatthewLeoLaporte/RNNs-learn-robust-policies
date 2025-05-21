from collections.abc import Callable
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree

from feedbax.bodies import SimpleFeedback
from feedbax.nn import NetworkState
from feedbax.task import SimpleReaches

from rnns_learn_robust_motor_policies.analysis.analysis import AbstractAnalysis, AnalysisDependenciesType, AnalysisInputData, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.analysis.fp_finder import (
    FixedPointFinder,
    fp_adam_optimizer,
    take_top_fps,
)
from rnns_learn_robust_motor_policies.types import LDict, TreeNamespace


class SteadyStateFPs(AbstractAnalysis):
    """ 
    """

    dependencies: ClassVar[AnalysisDependenciesType] = MappingProxyType(dict())
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = None
    fig_params: FigParamNamespace = DefaultFigParamNamespace()

    def compute(
        self,
        data: AnalysisInputData,
        **kwargs,
    ):
        pass


def get_endpoint_positions(task):
    trials = task.validation_trials
    return (
        trials.inits[lambda s: s.mechanics.effector].pos,
        trials.targets[lambda s: s.mechanics.effector.pos].value[:, -1]
    )


def get_initial_network_inputs(task):
    trials = task.validation_trials
    return jnp.concatenate([
        trials.inputs['context'][..., None],
        trials.inputs['effector_target'].pos,
        trials.inputs['effector_target'].vel,
    ], axis=-1)[:, 0, :]  # Index the first time step
    

def get_simple_reach_endpoint_fps(
    model: SimpleFeedback, 
    task: SimpleReaches, 
    loss_tol: float = 1e-6, 
    outlier_tol: float = 1.0,
    unique_tol: float = 0.025,
    stride_trials: int = 1,
    stride_candidates: int = 1,
    *, 
    key,
) -> tuple:
    """Returns the fixed points of the neural network for the following network inputs:
    
    1. When the target position and feedback position both indicate the goal position, 
       and both the target and feedback velocities are zero.
    2. When the target indicates the goal, feedback indicates the initial position,
       and both velocities are zero.
    """    
    key_eval, key_fps = jr.split(key)
        
    states = task.eval(model, key=key_eval)

    fp_optimizer = fp_adam_optimizer()
    fpfinder = FixedPointFinder(fp_optimizer)
    fpfind = partial(fpfinder.find_and_filter, outlier_tol=outlier_tol, unique_tol=unique_tol)
    
    # fp_candidates = jnp.reshape(states.net.hidden[::stride_trials], (-1, hidden_size))
    
    inits_pos, goals_pos = get_endpoint_positions(task)
    inputs = get_initial_network_inputs(task)
    
    # Construct (constant) network inputs for all trials for which we'll find FPs
    n_task_inputs = inputs.shape[-1]
    inputs_star_func = lambda fb_pos: ((
            jnp.zeros(
                (task.n_validation_trials, model.step.net.input_size)
            )
        ).at[:, :n_task_inputs].set(inputs)
    ).at[:, n_task_inputs : n_task_inputs + 2].set(fb_pos)
    
    # TODO: Could do a similar thing as in `get_traj_fps` and only use the candidates for the current trial
    @eqx.filter_jit
    def get_condition_fps(net, input_star, candidates, loss_tol, key):
        # Create the network step function with key bound
        def net_step(h):
            return net(input_star, h, key=key)
     
        return fpfind(
            net_step,
            candidates,
            loss_tol,
            key=key,
        )
    
    def get_all_conditions_fps(inputs_star, net, hidden_states, loss_tol, key_fps, stride_trials):
        return eqx.filter_vmap(
            get_condition_fps, 
            in_axes=(None, 0, 0, None, None),
        )(
            net, 
            inputs_star[::stride_trials],
            hidden_states[::stride_trials],
            loss_tol, 
            key_fps,
        )
        
    all_fpf_results = jax.tree_map(
        lambda inputs_star: get_all_conditions_fps(
            inputs_star, 
            model.step.net.hidden,
            states.net.hidden,
            loss_tol, 
            key_fps,
            stride_trials,
        ),
        {
            "goals-goals": inputs_star_func(goals_pos),
            "inits-goals": inputs_star_func(inits_pos),
        },
        is_leaf=lambda x: isinstance(x, tuple),
    )

    return states, all_fpf_results 


def get_fp_trajs(
    net: eqx.Module,
    net_states: NetworkState,
    loss_tol: float = 1e-6, 
    outlier_tol: float = 1.0,
    unique_tol: float = 0.025,
    *,
    key: PRNGKeyArray,
):
    """Returns fixed points along a trajectory of network input states, using the trajectory of 
    network hidden states as candidates for the optimization."""
    fp_optimizer = fp_adam_optimizer()
    fpfinder = FixedPointFinder(fp_optimizer)
    
    def get_state_fps(net, input, candidates, loss_tol, key):
        return fpfinder.find_and_filter(
            lambda h: net(input, h, key=key), 
            candidates, 
            loss_tol, 
            outlier_tol=outlier_tol, 
            unique_tol=unique_tol, 
            key=key,
        )    
    
    # (conditions, inputs/timesteps, candidates/timesteps, state)
    return eqx.filter_vmap(  
        eqx.filter_vmap(  
            get_state_fps, 
            in_axes=(None, 0, None, None, None), # over timesteps (of network inputs)
        ), 
        in_axes=(None, 0, 0, None, None) # over conditions
    )(
        net.hidden, 
        net_states.input, 
        net_states.hidden, 
        loss_tol, 
        key,
    )
    

def get_simple_reach_fps(
    model: SimpleFeedback, 
    task: SimpleReaches, 
    loss_tol: float = 1e-6, 
    outlier_tol: float = 1.0,
    unique_tol: float = 0.025,
    stride_trials: int = 1,
    *, 
    key: PRNGKeyArray,
    **kwargs,
) -> tuple:
    """Returns the fixed points of the neural network for the following network inputs:
    
    1. When the target position and feedback position both indicate the goal position, 
       and both the target and feedback velocities are zero.
    2. When the target indicates the goal, feedback indicates the initial position,
       and both velocities are zero.
    3. The actual network inputs across all timesteps of all the validation trials. 
    
    The validation trials are evaluated, and the resulting hidden states are used as candidates
    in each fixed point optimization. Whereas the number of candidates determines the number 
    of potential fixed points returned in cases 1 and 2, there is a factor of `task.n_steps` more
    potential fixed points returned in case 3, since it repeats the fixed point analysis for 
    each time step of the validation trials. 
    
    """
    states, all_fps = get_simple_reach_endpoint_fps(
        model, task, loss_tol, outlier_tol, unique_tol, stride_trials, key=key, **kwargs
    )

    # (conditions, inputs/timesteps, candidates/timesteps, state)
    all_fps['states'] = get_fp_trajs(
        model.step.net,
        states.net,
        loss_tol,
        outlier_tol,
        unique_tol,
        key=key,
    )
    
    return states, LDict.of("fp")(all_fps)


def get_simple_reach_first_fps(model, task, loss_tol, stride_trials=1, *, key):
    """Similar to `get_nn_fps`, but only returns the first fixed point for 
    each validation trial.
    
    Useful for networks where there is typically only one fixed point in 
    the operational region, e.g. simple reaching networks. 
    """
    states, all_fpf_results = get_simple_reach_fps(
        model, task, loss_tol, stride_trials=stride_trials, key=key
    )
    return states, take_top_fps(all_fpf_results, warn=False)