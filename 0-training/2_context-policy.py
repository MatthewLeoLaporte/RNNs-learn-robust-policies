#!/usr/bin/env python
# coding: utf-8

# ---
# jupyter: python3
# ---

# In[ ]:


NB_ID = "2"


# Training models for Part 2
 
## Environment setup


import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


from functools import partial
from typing import Any, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import numpy as np
import optax 
import plotly.graph_objects as go

import feedbax
from feedbax import (
    is_module,
    is_type,
    tree_unzip,
    tree_map_tqdm,
)
from feedbax.misc import where_func_to_labels
from feedbax.train import TaskTrainer
from feedbax.xabdeef.losses import simple_reach_loss

import rnns_learn_robust_motor_policies 
from rnns_learn_robust_motor_policies import PROJECT_SEED
from rnns_learn_robust_motor_policies.constants import INTERVENOR_LABEL
from rnns_learn_robust_motor_policies.database import (
    get_db_session,
    save_model_and_add_record,
)
from rnns_learn_robust_motor_policies.misc import log_version_info
from rnns_learn_robust_motor_policies.setup_utils import (
    get_readout_norm_loss,
)
from rnns_learn_robust_motor_policies.train_setup_part2 import (
    TrainingMethodLabel,
    setup_task_model_pair, 
)
from rnns_learn_robust_motor_policies.tree_utils import pp, subdict
from rnns_learn_robust_motor_policies.types import TaskModelPair
from rnns_learn_robust_motor_policies.train_setup import (
    concat_save_iterations,
    iterations_to_save_model_parameters,
    make_delayed_cosine_schedule,
)
from rnns_learn_robust_motor_policies.types import (
    TrainingMethodDict,
    TrainStdDict,
)


# Log the library versions and the feedbax commit ID, so they appear in any reports generated from this notebook.
version_info = log_version_info(
    jax, eqx, optax, git_modules=(feedbax, rnns_learn_robust_motor_policies)
)


### Initialize model database connection

db_session = get_db_session()


### Hyperparameters

disturbance_type: Literal['curl', 'constant'] = 'curl'  
feedback_delay_steps = 0
feedback_noise_std = 0.01
motor_noise_std = 0.01
hidden_size = 100
n_replicates = 5
n_steps = 100
dt = 0.05

n_batches_baseline = 0
n_batches_condition = 500
batch_size = 250
learning_rate_0 = 0.001
constant_lr_iterations = 0 # Number of initial training iterations to hold lr constant
cosine_annealing_alpha = 1.0  # Max learning rate factor decrease during cosine annealing 
weight_decay = 0

# Force the Frobenius norm of the readout weight matrix to be close (squared error) to this value
readout_norm_value = 2.0
readout_norm_loss_weight = 0.0

# TODO: Implement this for part 2!
n_scaleup_batches = 1000
intervention_scaleup_batches = (n_batches_baseline, n_batches_baseline + n_scaleup_batches)

# reset the optimizer state at these iterations
state_reset_iterations = jnp.array([])

# change which parameters are trained, after a given number of iterations
where_train = {
    0: lambda model: (
        model.step.net.hidden,
        model.step.net.readout, 
    ),
    # stop training the readout 
    # 1000: lambda model: model.step.net.hidden,
}

training_methods: list[TrainingMethodLabel] = ["bcs"]#, "pai-asf"]

p_perturbed = {
    "bcs": 0.5,
    # The rest don't do anything atm, even if they're <1
    "dai": 1.0,  
    "pai-asf": 1.0,  
}

# Define the disturbance amplitudes to train, depending on disturbance type
# NOTE: Only one of these disturbance types is trained per notebook run; see the parameters cell above
disturbance_stds = {
    # 'curl': [1.0],
    'curl': [0.0, 0.5, 1.0, 1.5],
    'constant': [0.0, 0.01, 0.02, 0.03, 0.04, 0.08, 0.16, 0.32],
}


### RNG setup
key = jr.PRNGKey(PROJECT_SEED)
key_init, key_train, key_eval = jr.split(key, 3)


## Set up models and tasks for the different training variants

task_model_pairs = TrainingMethodDict({
    method_label: jt.map(
        lambda disturbance_std: setup_task_model_pair(
            n_replicates=n_replicates,
            training_method=method_label,
            dt=dt,
            hidden_size=hidden_size,
            n_steps=n_steps,
            feedback_delay_steps=feedback_delay_steps,
            feedback_noise_std=feedback_noise_std,
            motor_noise_std=motor_noise_std,
            disturbance_type=disturbance_type,
            disturbance_std=disturbance_std,
            intervention_scaleup_batches=intervention_scaleup_batches,
            p_perturbed=p_perturbed,
            key=key_init,
        ),
        TrainStdDict(zip(
            disturbance_stds[disturbance_type], 
            disturbance_stds[disturbance_type],
        )),
    )
    for method_label in training_methods
})

# The task without training perturbations
# task_baseline = task_model_pairs[0].task


## Training setup

optimizer_class = partial(
    optax.adamw,
    weight_decay=weight_decay,
)

n_batches = n_batches_baseline + n_batches_condition
save_model_parameters = iterations_to_save_model_parameters(n_batches)

schedule = make_delayed_cosine_schedule(
    learning_rate_0, 
    constant_lr_iterations, 
    n_batches, 
    cosine_annealing_alpha,
) 

trainer = TaskTrainer(
    optimizer=optax.inject_hyperparams(optimizer_class)(
        learning_rate=schedule
    ),
    checkpointing=True,
)

readout_norm_loss = readout_norm_loss_weight * get_readout_norm_loss(readout_norm_value)
loss_func = simple_reach_loss() + readout_norm_loss


## Examine the distributions of field strengths in training batches

keys_example_trials = jr.split(key_train, batch_size)

example_batches = jt.map(
    lambda pair: jax.vmap(pair.task.get_train_trial_with_intervenor_params)(keys_example_trials),
    task_model_pairs,
    is_leaf=is_type(TaskModelPair),
)


# from feedbax.task import TaskTrialSpec

# def plot_curl_amplitudes(trial_specs):
#     fig = go.Figure(layout=dict(
#         width=500,
#         height=400,
#     ))
#     # Assume these are constant over each trial
#     amplitude, scale, active = (
#         trial_specs.intervene[INTERVENOR_LABEL].amplitude[:, 0],
#         trial_specs.intervene[INTERVENOR_LABEL].scale[:, 0],
#         trial_specs.intervene[INTERVENOR_LABEL].active[:, 0],
#     )
#     field_amp = active * scale * amplitude
#     fig.add_trace(
#         go.Histogram(
#             x=field_amp,
#             xbins=dict(
#                 start=-4, 
#                 end=4,
#                 size=0.3,
#             )
#         )
#     )
#     return fig
    
    
# field_amp_figs = jt.map(
#     plot_curl_amplitudes,
#     example_batches,
#     is_leaf=is_type(TaskTrialSpec)
# ) 


## Train the task-model pairs

train_params = dict(
    ensembled=True,
    loss_func=loss_func,
    where_train=where_train,
    batch_size=batch_size, 
    log_step=500,
    save_model_parameters=save_model_parameters,
    state_reset_iterations=state_reset_iterations,
    # disable_tqdm=True,
)

trained_models, train_histories = tree_unzip(tree_map_tqdm(
    partial(train_pair, trainer, n_batches, **train_params),
    task_model_pairs,
    label="Training all pairs",
    is_leaf=is_type(TaskModelPair),
))


## Save the models with their parameters on the final iteration


save_model_parameters_all = concat_save_iterations(
    save_model_parameters, 
    (n_batches_baseline, n_batches_condition),
)

where_train_strs = jt.map(where_func_to_labels, where_train)

training_hyperparameters = dict(
    learning_rate_0=learning_rate_0,
    constant_lr_iterations=constant_lr_iterations,
    cosine_annealing_alpha=cosine_annealing_alpha,
    weight_decay=weight_decay,
    n_batches=n_batches,
    n_batches_condition=n_batches_condition,
    n_batches_baseline=n_batches_baseline,
    batch_size=batch_size,
    save_model_parameters=save_model_parameters.tolist(),
    where_train_strs=where_func_to_labels(where_train[0]),
    state_reset_iterations=state_reset_iterations.tolist(),
    p_perturbed=p_perturbed,
)

model_hyperparameters = dict(
    n_replicates=n_replicates,
    hidden_size=hidden_size,
    feedback_delay_steps=feedback_delay_steps,
    feedback_noise_std=feedback_noise_std,
    motor_noise_std=motor_noise_std,
    dt=dt,
    n_steps=n_steps,
    disturbance_type=disturbance_type,
    # disturbance_std=disturbance_std,
    readout_norm_loss_weight=readout_norm_loss_weight,
    readout_norm_value=readout_norm_value,
    intervention_scaleup_batches=intervention_scaleup_batches,
    p_perturbed=p_perturbed,
)

train_histories_hyperparameters = dict(
    disturbance_stds=disturbance_stds[disturbance_type],
    n_batches=n_batches,
    batch_size=batch_size,
    n_replicates=n_replicates,
    where_train_strs=where_func_to_labels(where_train[0]),
    save_model_parameters=save_model_parameters.tolist(),
    readout_norm_loss_weight=readout_norm_loss_weight,
    readout_norm_value=readout_norm_value,
)

model_record = TrainingMethodDict({
    method_label: TrainStdDict({
        disturbance_std: save_model_and_add_record(
            db_session,
            origin=NB_ID,
            model=models,
            model_hyperparameters=model_hyperparameters | dict(
                disturbance_std=disturbance_std,
                training_method=method_label,
            ),
            other_hyperparameters=training_hyperparameters,
            train_history=train_histories,
            train_history_hyperparameters=train_histories_hyperparameters,
            version_info=version_info,
        )
        for disturbance_std, models in trained_models[method_label].items()
    })
    for method_label in training_methods
})

