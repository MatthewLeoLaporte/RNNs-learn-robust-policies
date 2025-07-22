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

from jax_cookbook import is_module
import jax_cookbook.tree as jtree
from jaxtyping import Array, Float, PRNGKeyArray
import numpy as np

from rlrmp.analysis import AbstractAnalysis, AnalysisInputData
from rlrmp.analysis.analysis import _DummyAnalysis, AnalysisDefaultInputsType, Data, DefaultFigParamNamespace, FigParamNamespace
from rlrmp.analysis.pca import StatesPCA
from rlrmp.misc import get_constant_input_fn
from rlrmp.analysis.state_utils import get_best_replicate, vmap_eval_ensemble
from rlrmp.tree_utils import take_replicate
from rlrmp.types import TreeNamespace
from rlrmp.types import LDict
from rlrmp.analysis.fps_tmp import (
    FixedPoints,
    SteadyStateJacobians,
    Jacobians,
    FPsInPCSpace,
)


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
        LDict.of('sisu')({
            sisu: (
                task_base.add_input(
                    name="sisu",
                    input_fn=get_constant_input_fn(
                        sisu, hps.model.n_steps, task_base.n_validation_trials,
                    ),
                ),
                models_base,  
                hps | dict(sisu=sisu),
            )
            for sisu in hps.sisu
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
STRIDE_FP_CANDIDATES = 16

def get_ss_rnn_input(input_size: int, sisu: float, pos: Float[Array, "2"]):
    input_star = jnp.zeros((input_size,))
    # Set target and feedback inputs to the same position
    input_star = input_star.at[1:3].set(pos)
    input_star = input_star.at[5:7].set(pos)
    return input_star.at[0].set(sisu)


def get_ss_rnn_func(sisu: float, rnn_cell: Module, pos: Float[Array, "2"], key: PRNGKeyArray):
    input_star = get_ss_rnn_input(pos, sisu, rnn_cell.input_size)
    def rnn_func(h):
        return rnn_cell(input_star, h, key=key)
    return rnn_func


#! After figuring this out, might be possible to reduce to a prep op on 
#! `funcs=Data.models(where=lambda model: model.step.net.hidden)`
class SteadyStateRNNFuncs(AbstractAnalysis):
    """Find steady-state RNN functions for the given RNN cell."""
    
    default_inputs: ClassVar[AnalysisDefaultInputsType] = MappingProxyType(dict())
    conditions: tuple[str, ...] = ()
    variant: Optional[str] = None
    fig_params: FigParamNamespace = DefaultFigParamNamespace()
    
    def compute(self, data: AnalysisInputData, **kwargs) -> TreeNamespace:
        """Compute the steady-state RNN functions."""
        models, states, hps = data.models, data.states, data.hps
        
        # 1. RNN cells vary in `models` by `train__pert__std`
        # 2. RNN funcs will vary with `train__pert__std`, `sisu`, and goal position
        
        return jt.map(lambda model: model.step.net.hidden, models, is_leaf=is_module)


DEPENDENCIES = {
    "states_pca": (
        StatesPCA(n_components=N_PCA, where_states=lambda states: states.net.hidden)
        .after_transform(get_best_replicate)
        .after_indexing(-2, np.arange(START_STEP, END_STEP), axis_label="timestep")
    ),
    "steady_state_rnn_funcs": (
        SteadyStateRNNFuncs()
    ),
    "steady_state_fps": (
        FixedPoints(
            custom_inputs=dict(
                funcs="steady_state_rnn_funcs",
                candidates=Data.states(
                    # FP initial conditions <- full hidden state trajectories
                    where=lambda states: states.net.hidden
                ),
            )
        )
        .after_transform(get_best_replicate)
        .after_transform(  
            # Collapse all axes except the state dimension 
            lambda states, *, hps_common: jnp.reshape(
                states,
                (-1, hps_common.train.model.hidden_size),
            )[::STRIDE_FP_CANDIDATES]
        )
    )
}


# State PyTree structure: ['sisu', 'train__pert__std']
# Array batch shape: (evals, replicates, reach conditions)
ANALYSES = {
    "fps_in_pc_space": (
        FPsInPCSpace(custom_inputs=dict(pca_results="states_pca"))
    ),
    "jacobians": (
        Jacobians(
            func_where=lambda model: model.step.net.hidden,
            inputs_where=lambda states: states.net.input,
            states_where=lambda states: states.net.hidden,
        )
        .map_figs_at_level("train__pert__std")
    ),
}


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

