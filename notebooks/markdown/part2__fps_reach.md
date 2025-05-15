---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: .venv
    language: python
    name: python3
---

```python
ANALYSIS_NAME = "part2.fps_reach"
```

# Analysis of fixed points during simple reaching

In each case, we should look at the following, across context inputs

1. project the fixed points into PC space
2. examine their Jacobian eigenspectra 

## Outline

### Find the structure of the steady (goal-goal) FPs

**TODO**: Do the structures (e.g. the goal-goal ring) change with the context input? (In principle the fixed points could remain in place, while the local dynamics change.)

### Find the structure of the initial unsteady (init-goal) FPs

Similarly, see how the structure of the initial fixed points changes

## Environment setup 

```python
%load_ext autoreload
%autoreload 2
```

```python
import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
```

```python
from collections.abc import Sequence
import functools
from functools import partial
from typing import Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from jaxtyping import Array, PyTree
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.colors as plc
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from feedbax import (
    is_module, 
    is_type,
    load, 
    tree_map_tqdm,
    tree_set_scalar,
    tree_stack,
    tree_struct_bytes,
    tree_take, 
    tree_take_multi,
    tree_unzip,
)
from feedbax.bodies import SimpleFeedbackState
from feedbax.intervene import (
    CurlField,
    FixedField,
    add_intervenors,
    schedule_intervenor,
)
from feedbax.misc import batch_reshape
import feedbax.plotly as fbp
from feedbax.task import (
    SimpleReaches, 
    TrialSpecDependency,
    centreout_endpoints,
)
from feedbax.xabdeef.losses import simple_reach_loss

from rnns_learn_robust_motor_policies import PROJECT_SEED
from rnns_learn_robust_motor_policies.colors import (
    COLORSCALES, 
    MEAN_LIGHTEN_FACTOR,
    get_colors_dicts,
)
from rnns_learn_robust_motor_policies.constants import (
    EVAL_REACH_LENGTH,
    INTERVENOR_LABEL,
    REPLICATE_CRITERION,
    WORKSPACE,
)
from rnns_learn_robust_motor_policies.database import (
    ModelRecord,
    get_db_session,
    get_model_record,
    add_evaluation,
    add_evaluation_figure,
    use_record_params_where_none,
)
from rnns_learn_robust_motor_policies.analysis.fp_finder import (
    FixedPointFinder,
    FPFilteredResults,
    fp_adam_optimizer,
    take_top_fps,
)
from rnns_learn_robust_motor_policies.analysis.fps import (
    get_simple_reach_first_fps,
)
from rnns_learn_robust_motor_policies.misc import (
    create_arr_df, 
    log_version_info,
)
from rnns_learn_robust_motor_policies.plot import (
    plot_eigvals_df,
    plot_fp_pcs,
)
from rnns_learn_robust_motor_policies.plot_utils import (
    PlotlyFigureWidget as PFW,
    figs_flatten_with_paths,
    figleaves,
    copy_fig_json,
)
from rnns_learn_robust_motor_policies.post_training import setup_replicate_info
from rnns_learn_robust_motor_policies.setup_utils import (
    get_base_task,
    query_and_load_model,
    set_model_noise,
    setup_models_only,
)
from rnns_learn_robust_motor_policies.state_utils import (
    vmap_eval_ensemble,
    get_pos_endpoints,
)
from rnns_learn_robust_motor_policies.tree_utils import (
    pp,
    subdict, 
    take_replicate,
)
from rnns_learn_robust_motor_policies.types import (
    ContextInputDict,
    PertAmpDict, 
    TrainingMethodDict,
    TrainStdDict,
)
```

### Colors setup

```python
conditions_colors = plc.sample_colorscale('phase', np.linspace(0, 1, n_directions))
conditions_colors_tuples = plc.convert_colors_to_same_type(
    plc.sample_colorscale('phase', np.linspace(0, 1, n_directions)), 
    colortype='tuple'
)[0]
```

## Define tasks

### Define the disturbances

```python
# Evaluate only a single amplitude, for now;
# we want to see variation over the context input
if disturbance_type == 'curl':  
    def disturbance(amplitude):
        return CurlField.with_params(amplitude=amplitude)    
        
elif disturbance_type == 'constant':   
    def disturbance(amplitude):            
        return FixedField.with_params(
            scale=amplitude,
            field=orthogonal_field,  
        ) 
          
else:
    raise ValueError(f"Unknown disturbance type: {disturbance_type}")
```

### Set up the base task 

See notebook 1-2a for some explanation of the parameter choices here.

```python
# Also add the intervenors to the trained models
task_base, all_models = schedule_intervenor(
    get_base_task(
        model_info_0.n_steps,
        validation_params=dict(
            eval_grid_n=n_grid,
            eval_n_directions=n_directions,
            eval_reach_length=EVAL_REACH_LENGTH,
        )
    ),
    models_load,
    lambda model: model.step.mechanics,
    disturbance(disturbance_amplitude),
    label=INTERVENOR_LABEL,
    default_active=False,
)
```

### Set up variants with different context inputs

```python
def get_context_input_func(x, n_steps, n_trials):
    return lambda trial_spec, key: (
        jnp.full((n_trials, n_steps), x, dtype=float)
    )

all_tasks = ContextInputDict({
    context_input: eqx.tree_at(
        lambda task: task.input_dependencies,
        task_base, 
        {
            'context': TrialSpecDependency(
                get_context_input_func(
                    context_input, 
                    model_info_0.n_steps - 1, 
                    task_base.n_validation_trials
                ),
            ),
        },
    )
    for context_input in context_inputs
})
```

### Assign some things for convenient reference

```python
example_task = jt.leaves(all_tasks, is_leaf=is_module)[0]

trial_specs = jt.map(lambda task: task.validation_trials, example_task, is_leaf=is_module)

pos_endpoints = jt.map(get_pos_endpoints, trial_specs, is_leaf=is_module)
```

## Get the FPs of all models, for all values of the context input

```python
stride_trials = 1  # only do the FP analysis for every Nth trial (reach direction)
loss_tol = 1e-6  # threshold criterion for FP optimization

def get_all_states_and_fps():

    states, fps = tree_unzip(tree_map_tqdm(
        lambda model: LDict.of("context_input")({
            context_input: get_simple_reach_first_fps(
                model, task, loss_tol, stride_trials=stride_trials, key=key_eval
            )
            for context_input, task in all_tasks.items()
        }),
        all_models,
        is_leaf=is_module,
    ))
    return states, fps
```

```python
# all_states_and_fps_shape = eqx.filter_eval_shape(get_all_states_and_fps)

# print(
#     f"{tree_struct_bytes(all_states_and_fps_shape) / 1e9:.3f} GB"
#     " of memory estimated to store all states."
# )
```

```python
# with jax.checking_leaks():
#     all_states_and_fps_shape = eqx.filter_eval_shape(get_all_states_and_fps)
```

```python
all_states, all_fps = get_all_states_and_fps()
```

```python
all_fps = jt.map(jnp.squeeze, all_fps)
```

```python
all_hidden = jt.map(
    lambda states: states.net.hidden,
    all_states,
    is_leaf=is_module,
)
```

## Select a model to analyze, and perform PCA on its hidden states

```python
training_method = 'pai-asf'
train_std = 1.0
```

Stack the context inputs into the first array dimensions.

```python
fps = tree_stack(all_fps[training_method][train_std].values())
hidden = tree_stack(all_hidden[training_method][train_std].values())
```

Check if there are any NaN values. In particular, it seems like occasionally, no FPS will be found (i.e. the values will be NaN) for a small number of reach condition and context input combinations. 

```python
nan_fp_idxs = jt.map(lambda arr: np.where(np.isnan(arr)), fps)
pp(nan_fp_idxs)
```

Perform PCA on all the hidden states together, for all context inputs and reach directions.

```python
# Calculate the PCs starting midway through the trial, to prefer the goal plane.
# This is just to make the plots clearer; statistically it might not be as good. 
i_step_pca_start = 50
n_steps = model_info_0.n_steps

hidden_for_pca = hidden[..., i_step_pca_start - n_steps:, :].reshape(-1, hidden_size)
pca2 = PCA(n_components=2).fit(hidden_for_pca)
pca = PCA(n_components=30).fit(hidden_for_pca)

# Convenience for doing PC projection of all batched arrays in a PyTree
pca_transform = lambda x: jt.map(batch_reshape(pca.transform), x)
```

```python
hidden_pc = pca_transform(hidden)

fps_pc = pca_transform(fps)
```

```python
readout_weights_pc = pca_transform(readout_weights[training_method][train_std])
```

## Visualize FPs

### Compare goals-goals and inits-goals FPs across context inputs

```python
pcs_plot = slice(0, 3)

fp_alpha = 0.4
stride_plot = 8

fig = go.Figure(
    layout=dict(
        width=800, 
        height=600,
        margin=dict(l=10, r=10, t=0, b=10),
        legend=dict(
            yanchor="top", 
            y=0.9, 
            xanchor="right", 
        ),
    ), 
)
fig = fbp.trajectories_3D(
    fps_pc['goals-goals'][..., pcs_plot], 
    colors=list(context_input_colors_dark.values()),
    mode='markers', 
    marker_size=4, 
    endpoint_symbol=None,
    name="Goals-goals FPs",
    axis_labels=('PC1', 'PC2', 'PC3'),
    fig=fig, 
    marker_symbol="circle-open",
)
fig = fbp.trajectories_3D(
    fps_pc['inits-goals'][..., pcs_plot], 
    colors=list(context_input_colors_dark.values()),
    mode='markers', 
    marker_size=4, 
    endpoint_symbol=None,
    name="Inits-goals FPs",
    axis_labels=('PC1', 'PC2', 'PC3'),
    fig=fig, 
    marker_symbol='circle',
)


fig.add_traces(
    readout_vector_traces(
        readout_weights_pc[..., pcs_plot],
        jnp.tile(hidden_pc[context_idx, 0, 0, pcs_plot], (2, 1)),
    ),
)

PFW(fig).show()
```

### Plotting functions

```python
from jaxtyping import Array, Float

def plot_hidden_and_fp_trajectories_3D(
    fp_trajs_pc: Float[Array, 'curves index pcs=3'],
    hidden_pc: Float[Array, 'curves index pcs=3'],
    colors: Sequence,  # len curves
    fp_alpha: float = 0.4,
    stride_plot: int = 1,
    marker_size: float = 2,
    axis_labels: tuple[str, str, str] = ('PC1', 'PC2', 'PC3'),
):
    fig = go.Figure(
        layout=dict(
            width=800, 
            height=600,
            margin=dict(l=10, r=10, t=0, b=10),
            legend=dict(
                yanchor="top", 
                y=0.9, 
                xanchor="right", 
            ),
            scene_aspectmode='data',
        ), 
    )
    fig = fbp.trajectories_3D(
        fp_trajs_pc[::stride_plot], 
        colors=colors[::stride_plot], 
        mode='markers', 
        marker_size=marker_size, 
        marker_opacity=fp_alpha,
        endpoint_symbol='square-open',
        name="Local FP",
        axis_labels=axis_labels,
        fig=fig, 
    )
    fig = fbp.trajectories_3D(
        hidden_pc[::stride_plot], 
        colors=colors[::stride_plot], 
        mode='lines', 
        line_width=2,
        endpoint_symbol='diamond-open', 
        name='Reach trajectory',
        axis_labels=axis_labels,
        fig=fig,
    )
    
    return fig
```

```python
def readout_vector_traces(
    readout_weights_pc: Float[Array, 'out=2 pcs=3'], 
    vector_start_pc: Optional[Float[Array, 'out=2 pcs=3']] = None,
    colors: tuple = ('#FF0000', '#0000FF'),
    scale: float = 0.25,
):
    fig.update_layout(
        legend2=dict(
            title='Readout<br>components',
            itemsizing='constant',
            y=0.45,
        ),
    )

    traces = []

    if vector_start_pc is None:
        vector_start_pc = np.zeros_like(readout_weights_pc)

    for j, readout_label in enumerate(('x', 'y')):
        start = vector_start_pc[j]
        end = vector_start_pc[j] + scale * readout_weights_pc[j]

        # # Interleave start and end points with None for multiple disconnected lines
        x = np.column_stack((start[..., 0], end[..., 0], np.full_like(start[..., 0], None))).ravel()
        y = np.column_stack((start[..., 1], end[..., 1], np.full_like(start[..., 1], None))).ravel()
        z = np.column_stack((start[..., 2], end[..., 2], np.full_like(start[..., 2], None))).ravel()

        traces.append(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                line=dict(width=10),
                showlegend=True,
                name=readout_label,
                line_color=colors[j],
                legend="legend2",
            )
        )
    
    return traces
```

### Plot hidden and FP trajectories for a single context input

```python
pcs_plot = slice(0, 3)

context = 0
context_idx = context_inputs.index(context)
stride_plot = 6

fig = plot_hidden_and_fp_trajectories_3D(
    fps_pc['states'][context_idx, ..., pcs_plot],
    hidden_pc[context_idx, ..., pcs_plot],
    colors=conditions_colors,
    stride_plot=stride_plot,
)

fig.add_traces(
    readout_vector_traces(
        readout_weights_pc[..., pcs_plot],
        jnp.tile(hidden_pc[context_idx, 0, 0, pcs_plot], (2, 1)),
    ),
)

PWF(fig).show()
```

#### Corresponding effector trajectories

```python
states = all_states[training_method][train_std][context]

where_plot = lambda states: (
    states.mechanics.effector.pos,
    states.mechanics.effector.vel,
    states.efferent.output,
)

fig = fbp.trajectories_2D(
    jt.map(lambda arr: arr[::stride_plot], where_plot(states)),
    var_labels=('Position', 'Velocity', 'Control force'),
    axes_labels=('x', 'y'),
    legend_title='Reach direction',
)
PWF(fig).show()
```

#### 2D

```python
pcs = np.array([1, 2])

dimtoend = lambda x: jnp.moveaxis(x, 0, -1)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')

for i, color in enumerate(conditions_colors_tuples[::stride_plot]):
    ax.plot(
        *dimtoend(hidden_pc[context_idx, ::stride_plot, ..., pcs])[i].T, 
        color=color,
    )
    ax.plot(
        *dimtoend(fps_pc['goals-goals'][context_idx, ::stride_plot, ..., pcs])[i].T, 
        'o', 
        color=color, 
        markersize=10, 
        markerfacecolor="None", 
        markeredgewidth=3,
    )
    ax.plot(
        *dimtoend(fps_pc['states'][context_idx, ::stride_plot, ..., pcs])[i].T,
        'o',
        color=color,
        markersize=2,
        markerfacecolor=color,
    )
    
    
aa = jnp.pad(readout_weights_pc[..., pcs][None], ((1,0), (0,0), (0,0))).T
ax.plot(*aa[0], c='r')
ax.plot(*aa[1], c='b')

ax.set_xlabel(f"PC{pcs[0] + 1}")
ax.set_ylabel(f"PC{pcs[1] + 1}")
plt.show()
```

### Compare a single reach direction's FP trajectory, across context inputs

**TODO**. For both baseline and disturbance conditions. 

```python
pcs_plot = slice(0, 3)

direction_idx = 0

fig = plot_hidden_and_fp_trajectories_3D(
    fps_pc['states'][:, direction_idx, ..., pcs_plot],
    hidden_pc[:, direction_idx, ..., pcs_plot],
    colors=list(context_input_colors_dark.values()),
    # stride_plot=1,
)

fig.add_traces(
    readout_vector_traces(
        readout_weights_pc[..., pcs_plot],
        jnp.tile(hidden_pc[0, direction_idx, 0, pcs_plot], (2, 1)),
    ),
)

PFW(fig).show()
```

### Compare average aligned FP trajectories, across context inputs

TODO. I'm not sure this makes sense, since we need to align them in the high-dimensional space. But perhaps there is a principled way to do it; e.g. by rotating around (which point in?) the readout plane

