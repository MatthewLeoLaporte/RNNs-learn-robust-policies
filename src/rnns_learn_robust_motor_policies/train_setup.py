from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array 
import numpy as np
import optax


"""
Define the training iterations on which to retain the model weights:
Every iteration until iteration 10, then every 10 until 100, every 100 until 1000, etc.
"""
def iterations_to_save_model_parameters(n_batches):
    save_iterations = jnp.concatenate([jnp.array([0])] + [
        jnp.arange(10**i, 10**(i+1), 10**i)
        for i in range(0, int(np.log10(n_batches)) + 1)
    ])
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