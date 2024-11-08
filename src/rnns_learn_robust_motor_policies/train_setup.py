import jax.numpy as jnp
import numpy as np


"""
Define the training iterations on which to retain the model weights:
Every iteration until iteration 10, then every 10 until 100, every 100 until 1000, etc.
"""
def iterations_to_save_model_parameters(n_batches):
    return jnp.concatenate([jnp.array([0])] + [
        jnp.arange(10**i, 10**(i+1), 10**i)
        for i in range(0, int(np.log10(n_batches)))
    ])