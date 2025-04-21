"""
Analysis of fixed point (FP) structure of the network at steady state.

Includes eigendecomposition of the linearization (Jacobian).

Steady state FPs are the "goal-goal" FPs; i.e. the point mass is at the goal, so the
network will stabilize on some hidden state that outputs a constant force that does not change the
position of the point mass on average.

This contrasts with non-steady-state FPs, which correspond to network outputs which should
cause the point mass to move, and thus the network's feedback input (and thus FP) to
change.
"""

from collections.abc import Callable, Sequence
from types import MappingProxyType, SimpleNamespace
from typing import ClassVar, Optional

import equinox as eqx
from equinox import Module
import jax.numpy as jnp
import jax.tree as jt

from feedbax.task import TrialSpecDependency
from jax_cookbook import is_module
import jax_cookbook.tree as jtree
import numpy as np

from rnns_learn_robust_motor_policies.analysis import AbstractAnalysis
from rnns_learn_robust_motor_policies.analysis.analysis import _DummyAnalysis, DefaultFigParamNamespace, FigParamNamespace
from rnns_learn_robust_motor_policies.analysis.pca import StatesPCA
from rnns_learn_robust_motor_policies.misc import get_constant_input
from rnns_learn_robust_motor_policies.analysis.state_utils import get_best_replicate_states, vmap_eval_ensemble
from rnns_learn_robust_motor_policies.tree_utils import take_replicate
from rnns_learn_robust_motor_policies.types import TreeNamespace
from rnns_learn_robust_motor_policies.types import LDict


ID = "2-4"


"""Specify any additional colorscales needed for this analysis. 
These will be included in the `colors` kwarg passed to `AbstractAnalysis` methods
"""
COLOR_FUNCS: dict[str, Callable[[TreeNamespace], Sequence]] = dict(
)


def setup_eval_tasks_and_models(task_base: Module, models_base: LDict[float, Module], hps: TreeNamespace):
    """Define a task where the point mass is already at the goal.

    This is similar to `feedback_perts`, but without an impulse.
    """
    all_tasks, all_models, all_hps = jtree.unzip(
        LDict.of("context_input")({
            context_input: (
                eqx.tree_at(
                    lambda task: task.input_dependencies,
                    task_base,
                    {
                        'context': TrialSpecDependency(get_constant_input(
                            context_input, hps.model.n_steps, task_base.n_validation_trials,
                        ))
                    },
                ),
                models_base,  
                hps | dict(context_input=context_input),
            )
            for context_input in hps.context_input
        })
    )
    # Provides any additional data needed for the analysis
    extras = SimpleNamespace()  
    
    return all_tasks, all_models, all_hps, extras


# Unlike `feedback_perts`, we don't need to vmap over impulse amplitude 
eval_func: Callable = vmap_eval_ensemble
 
# goals_pos = task.validation_trials.targets["mechanics.effector.pos"].value[:, -1]


N_PCA = 50
START_STEP = 0
END_STEP = 100


# State PyTree structure: ['context_input', 'train__pert__std']
# Array batch shape: (evals, replicates, reach conditions)
ALL_ANALYSES = [
    (
        StatesPCA(n_components=N_PCA, where_states=lambda states: states.net.hidden)
        .after_transform(get_best_replicate_states)
        .after_indexing(-2, np.arange(START_STEP, END_STEP), axis_label="timestep")
    ),

]


#! Get readout weights in PC space, for plotting
# all_readout_weights = exclude_nan_replicates(jt.map(
#     lambda model: model.step.net.readout.weight,
#     all_models,
#     is_leaf=is_module,
# ))

# all_readout_weights_pc = batch_transform(all_readout_weights)


"""
If we decide to aggregate over multiple replicates rather than taking only the best one,
it will be necessary to exclude the NaN replicates prior to performing sklearn PCA.
The following are the original functions I used for that purpose.
"""
def exclude_nan_replicates(tree, replicate_info, exclude_underperformers_by='best_total_loss'):
    return jt.map(
        take_replicate,
        replicate_info['included_replicates'][exclude_underperformers_by],
        tree,
    )

def take_non_nan(arr, axis=1):
    # Create tuple of axes to reduce over (all axes except the specified one)
    reduce_axes = tuple(i for i in range(arr.ndim) if i != axis)
    has_nan = jnp.any(jnp.isnan(arr), axis=reduce_axes)
    valid_cols = jnp.where(~has_nan)[0]
    return jnp.take(arr, valid_cols, axis=axis)