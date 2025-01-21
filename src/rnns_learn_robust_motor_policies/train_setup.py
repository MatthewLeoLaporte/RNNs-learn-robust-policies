from collections.abc import Sequence
from functools import partial
from typing import Optional

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray
import numpy as np
import optax

from feedbax import tree_concatenate
from feedbax.loss import AbstractLoss
from feedbax.task import AbstractTask
from feedbax.train import TaskTrainer
from feedbax.xabdeef.losses import simple_reach_loss

from rnns_learn_robust_motor_policies.loss import get_readout_norm_loss
from rnns_learn_robust_motor_policies.types import TaskModelPair


"""
Define the training iterations on which to retain the model weights:
Every iteration until iteration 10, then every 10 until 100, every 100 until 1000, etc.
"""
def iterations_to_save_model_parameters(n_batches):
    save_iterations = jnp.concatenate([jnp.array([0])] + 
        [
            jnp.arange(10 ** i, 10 ** (i + 1), 10 ** i)
            for i in range(0, int(np.log10(n_batches)) + 1)
        ]
    )
    return save_iterations[save_iterations < n_batches]
    

def make_delayed_cosine_schedule(init_lr, constant_steps, total_steps, alpha=0.001):
    """Returns an Optax schedule that starts with constant learning rate, then cosine anneals."""
    constant_schedule = optax.constant_schedule(init_lr)
    
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=init_lr,
        decay_steps=max(0, total_steps - constant_steps),
        alpha=alpha,
    )
    return optax.join_schedules(
        schedules=[constant_schedule, cosine_schedule],
        boundaries=[constant_steps]
    )
    

def concat_save_iterations(iterations: Array, n_batches_seq: Sequence[int]):
    total_batches = np.cumsum([0] + list(n_batches_seq))
    return jnp.concatenate([
        iterations[iterations < n] + total for n, total in zip(n_batches_seq, total_batches)
    ])
    

def train_pair(
    trainer: TaskTrainer, 
    n_batches: int,
    pair: TaskModelPair, 
    task_baseline: Optional[AbstractTask] = None,
    n_batches_baseline: int = 0,
    *,
    key: PRNGKeyArray,
    **kwargs,
):   
    key0, key1 = jr.split(key, 2)
    
    if n_batches_baseline > 0 and task_baseline is not None:
        pretrained, pretrain_history, opt_state = trainer(
            task_baseline,
            pair.model,
            n_batches=n_batches_baseline, 
            run_label="Baseline training",
            key=key0,
            **kwargs,
        )
    else: 
        pretrained = pair.model
        pretrain_history = None
        opt_state = None
    
    trained, train_history, _ = trainer(
        pair.task, 
        pretrained,
        opt_state=opt_state,
        n_batches=n_batches, 
        idx_start=n_batches_baseline,
        run_label="Condition training",
        key=key1,
        **kwargs,
    )
    
    if pretrain_history is None:
        train_history_all = train_history
    else:
        train_history_all = tree_concatenate([pretrain_history, train_history])
    
    return trained, train_history_all


def train_setup(
    hyperparams: dict,
) -> tuple[TaskTrainer, AbstractLoss]:
    """Given hyperparameters, """
    optimizer_class = partial(
        optax.adamw,
        weight_decay=hyperparams['weight_decay'],
    ) 

    schedule = make_delayed_cosine_schedule(
        hyperparams['learning_rate_0'], 
        hyperparams['constant_lr_iterations'], 
        hyperparams['n_batches_baseline'] + hyperparams['n_batches_condition'], 
        hyperparams['cosine_annealing_alpha'],
    ) 

    trainer = TaskTrainer(
        optimizer=optax.inject_hyperparams(optimizer_class)(
            learning_rate=schedule,
        ),
        checkpointing=True,
    )
    
    loss_func = simple_reach_loss()
    
    if all(k in hyperparams for k in ('readout_norm_loss_weight', 'readout_norm_value')):
        readout_norm_loss = (
            hyperparams['readout_norm_loss_weight'] 
            * get_readout_norm_loss(hyperparams['readout_norm_value'])
        )
        loss_func = loss_func + readout_norm_loss
    
    return trainer, loss_func