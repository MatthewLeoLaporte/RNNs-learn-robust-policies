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

---
jupyter: python3
format:
  html:
    toc: true 
execute:
  echo: false
---


I'm going to try something here I haven't before: performing an evaluation/analysis as in
`scripts/run_analysis.py`, but then operate on the results in a notebook.

```python
# Associate this notebook with a particular analysis module to load and run.
ANALYSIS_ID = "2-4"
```

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
import numpy as np
import plotly
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
from feedbax.misc import batch_reshape
import feedbax.plotly as fbp
from feedbax.task import centreout_endpoints
import jax_cookbook.tree as jtree

from rnns_learn_robust_motor_policies.analysis.execution import run_analysis_module
from rnns_learn_robust_motor_policies.analysis.fp_finder import (
    FixedPointFinder,
    FPFilteredResults,
    fp_adam_optimizer,
    take_top_fps,
)
from rnns_learn_robust_motor_policies.analysis.state_utils import get_best_replicate, exclude_bad_replicates
from rnns_learn_robust_motor_policies.colors import (
    COLORSCALES, 
    MEAN_LIGHTEN_FACTOR,
)
from rnns_learn_robust_motor_policies.config import PRNG_CONFIG
from rnns_learn_robust_motor_policies.database import add_evaluation_figure
from rnns_learn_robust_motor_policies.misc import create_arr_df, take_non_nan, lohi
from rnns_learn_robust_motor_policies.plot import (
    plot_eigvals_df,
    plot_fp_pcs,
)
from rnns_learn_robust_motor_policies.plot_utils import (
    figs_flatten_with_paths,
    figleaves,
)
from rnns_learn_robust_motor_policies.tree_utils import (
    first,
    first_shape as fs,
    pp,
    subdict, 
    take_replicate,
    tree_level_labels,
)
from rnns_learn_robust_motor_policies.types import LDict, TreeNamespace
```

## Setup task-model pairs, evaluate states, and perform PCA on activities

```python
key = jr.PRNGKey(PRNG_CONFIG.seed)
_, _, key_eval = jr.split(key, 3)

data, common_data, all_results, _ = run_analysis_module(ANALYSIS_ID, key=key)
```

```python
# Expand out the data/results for easier reference
models_ = data.models["full"] 
tasks = data.tasks["full"]
states_ = data.states["full"] 
hps = data.hps["full"]
# extras = data.extras 

hps_common = common_data["hps_common"]
replicate_info = common_data["replicate_info"]
trial_specs = common_data["trial_specs"]
colors = common_data["colors"]
colorscales = common_data["colorscales"]

pca = all_results[0].pca
pca_batch_transform = all_results[0].batch_transform
```

### Other useful stuff

```python
example_task = jt.leaves(tasks, is_leaf=is_module)[0]
goals_pos = example_task.validation_trials.targets["mechanics.effector.pos"].value[:, -1]

origin_idx = hps_common.task.full.eval_grid_n ** 2 // 2
```

#### Readout weight vectors in PC space

```python
all_readout_weights = exclude_bad_replicates(
    jt.map(
        lambda model: model.step.net.readout.weight,
        models_[0],  # Weights do not depend on context input
        is_leaf=is_module,
    ), 
    replicate_info=replicate_info, 
    axis=0,
)

all_readout_weights_pc = pca_batch_transform(all_readout_weights)
```

### Rearrange the LDict levels

This is because the existing code expects the context input level to be on the inside

```python
from rnns_learn_robust_motor_policies.tree_utils import ldict_level_to_bottom, ldict_level_to_top, move_ldict_level_above

#! TODO: Could avoid this if we use stack-and-vmap instead of needing to `jt.map` over LDict levels
#! in the fixed point finding functions
models, states = [
    ldict_level_to_bottom("context_input", tree, is_leaf=is_module)
    for tree in (models_, states_)
]

rnn_funcs = jt.map(
    lambda model: model.step.net.hidden,
    models,
    is_leaf=is_module,
)
```

## Get steady-state fixed points

These are the "goal-goal" FPs; i.e. the point mass is at the goal, so the network will stabilize on some hidden state that outputs a constant force that does not change the position of the point mass on average.

This contrasts with non-steady-state fixed points, which correspond to network outputs which should cause the point mass to move, and thus the network's feedback input (and thus fixed point) to change.


#### Get the readout weight vectors in PC space


### Instantiate the fixed point (FP) finder

```python
# Fixed point optimization hyperparameters
fp_tol = 1e-6  # Used for both fp_tol and opt_stop_tol
fp_num_batches = 10000         # Total number of batches to train on.
# fp_batch_size = 128          # How many examples in each batch
fp_step_size = 0.2          # initial learning rate
fp_decay_factor = 0.9999     # decay the learning rate this much
fp_decay_steps = 1           #
fp_adam_b1 = 0.9             # Adam parameters
fp_adam_b2 = 0.999
fp_adam_eps = 1e-5
fp_opt_print_every = 200   # Print training information during optimziation every so often

# Fixed point finding thresholds and other HPs
fp_noise_var = 0.0      # Gaussian noise added to fixed point candidates before optimization.
# fp_opt_stop_tol = 0.00001  # Stop optimizing when the average value of the batch is below this value.
# fp_tol = 0.00001        # Discard fps with squared speed larger than this value.
fp_unique_tol = 0.025   # tolerance for determination of identical fixed points
fp_outlier_tol = 1.0    # Anypoint whos closest fixed point is greater than tol is an outlier.
```

```python
fp_optimizer = fp_adam_optimizer(
    learning_rate=fp_step_size, 
    decay_steps=fp_decay_steps, 
    decay_rate=fp_decay_factor, 
    b1=fp_adam_b1, 
    b2=fp_adam_b2, 
    eps=fp_adam_eps, 
)

fpfinder = FixedPointFinder(fp_optimizer)

fpf_func = partial(
    fpfinder.find_and_filter,
    outlier_tol=fp_outlier_tol,
    unique_tol=fp_unique_tol,    
    key=key_eval,
)
```

### Define functions to calculate FPs

```python
def get_ss_network_input_with_context(pos, context, rnn_cell):
    input_star = jnp.zeros((rnn_cell.input_size,))
    # Set target and feedback inputs to the same position
    input_star = input_star.at[1:3].set(pos) 
    input_star = input_star.at[5:7].set(pos)
    return input_star.at[0].set(context)
    

def get_ss_rnn_func_at_context(pos, context, rnn_cell):
    input_star = get_ss_network_input_with_context(pos, context, rnn_cell)
    def rnn_func(h):
        return rnn_cell(input_star, h, key=key_eval)
    return rnn_func


def get_ss_rnn_fps(pos, rnn_cell, candidate_states, context):
    fps = fpf_func(
        get_ss_rnn_func_at_context(pos, context, rnn_cell), 
        candidate_states, 
        fp_tol,
    )
    return fps
```

```python
from collections.abc import Callable
from typing import Any

def multi_vmap(
    func: Callable, 
    in_axes_sequence: Sequence[PyTree[int | Callable[[Any], int] | None]],
    vmap_func: Callable = eqx.filter_vmap,
):
    """Given a sequence of `in_axes`, construct a nested vmap of `func`."""
    func_v = func
    for ax in in_axes_sequence:
        func_v = vmap_func(func_v, in_axes=ax)
    return func_v

@eqx.filter_jit
def get_ss_fps_at_positions(
    positions: float | Array, 
    all_funcs: PyTree[Callable, 'T'], 
    all_states: PyTree[LDict[float, SimpleFeedbackState], 'T'],
    context_inputs: Sequence[float],
    stride_candidates: int = 16,
):
    """For each of one or more workspace positions, find the respective fixed point of the RNN
    whose inputs indicate it's reached a goal at that position.
    
    Repeat over a range of fixed values of the RNN's context input.
    
    Exclude any replicate for which no fixed point meeting the acceptance criteria was found, 
    for at least one context input.
    """
    if isinstance(positions, Array) and len(positions.shape) == 2:
        # get_fps_func = eqx.filter_vmap(
        #     eqx.filter_vmap(
        #         get_ss_rnn_fps, 
        #         in_axes=(0, None, None, None),  # Over grid positions
        #     ),
        #     in_axes=(None, 0, 0, None),  # Over replicates
        # )
        get_fps_func = multi_vmap(
            get_ss_rnn_fps,
            in_axes_sequence=(
                (None, 0, 0, None),  # Over replicates
                (0, None, None, None),  # Over grid positions
            ),
        )
    else:
        get_fps_func = eqx.filter_vmap(
            get_ss_rnn_fps,
            in_axes=(None, 0, 0, None),  # Over replicates
        )
        
    candidates = jt.map(
        lambda states: jnp.reshape(
            states.net.hidden, 
            (hps_common.train.model.n_replicates, -1, hps_common.train.model.hidden_size),
        )[:, ::stride_candidates],
        all_states,
        is_leaf=is_module,
    )
    
    # tree_map_tqdm isn't helpful here since the calls are async 
    return jt.map(
        lambda func, candidates_by_context: LDict.of('context_input')({
            context_input: get_fps_func(
                positions,
                first(func, is_leaf=is_module),
                candidates_by_context[context_input],
                context_input,
            )
            for context_input in context_inputs
        }),
        all_funcs, candidates,
        is_leaf=LDict.is_of('context_input'),
    )
    
    
def process_fps(all_fps):
    """Only keep FPs/replicates that meet criteria."""
    n_fps_meeting_criteria = jt.map(
        lambda fps: fps.counts['meets_all_criteria'], 
        all_fps, 
        is_leaf=is_type(FPFilteredResults),
    )

    satisfactory_replicates = jt.map(
        lambda n_matching_fps_by_context: jnp.all(
            jnp.stack(jt.leaves(n_matching_fps_by_context), axis=0), 
            axis=0,
        ),
        n_fps_meeting_criteria,
        is_leaf=LDict.is_of('context_input'),
    )

    # NOTE: We aren't actually indexing out `fps[mask]` here, since we want to be able to vmap 
    # the replicate dimension here, with that of the models, when calculating the Jacobians. 
    # Later, we can exclude replicates from the final results based on `satisfactory_replicates`.

    all_top_fps = take_top_fps(all_fps, n_keep=6)
    
    # Average over the top fixed points, to get a single one for each included replicate and 
    # control input.
    fps_final = jt.map(
        lambda top_fps_by_context, mask: jt.map(
            lambda fps: jnp.nanmean(fps, axis=-2),  
            top_fps_by_context,
            is_leaf=is_type(FPFilteredResults),
        ),
        all_top_fps, satisfactory_replicates,
        is_leaf=LDict.is_of('context_input'),
    )
    
    return fps_final, all_top_fps, n_fps_meeting_criteria, satisfactory_replicates
```

### Get fixed points at a grid of steady-state positions

```python
fp_results = get_ss_fps_at_positions(
    goals_pos, rnn_funcs, states, hps_common.context_input,
)
```

```python
fps_grid, all_top_fps_grid, n_fps_meeting_criteria_grid, satisfactory_replicates_grid = process_fps(fp_results)
# del fp_results
```

```python
fps_grid = jt.map(
    lambda x: jnp.moveaxis(x, 0, 1),
    fps_grid,
)
```

## Examine the steady-state grid FP structure in PC space

How much do the fixed points vary across workspace positions? 

Is there translation invariance, or at the other extreme, does every steady-state point correspond to a different zero-force fixed point?

```python
fps_grid 
```

```python
# Note that `take_non_nan` excludes grid points where no fixed point was found;
# if we do this without first doing `exclude_nan_replicates`, then we'll definitely
# find a NaN for every index in axis 1, and this PyTree will be empty. 

fps_grid_pre_pc = jt.map(
    lambda fps: take_non_nan(fps, axis=1),
    exclude_bad_replicates(fps_grid, replicate_info=replicate_info),
)
```

```python
#!
#! Why does it appear here that there are no fps for e.g. std=1.0 & context=-3,
#! whereas in the Jacobian eigendecomposition we see that there are valid eigenvalues in these cases?
jt.map(lambda x: x.shape, fps_grid_pre_pc)
```

```python
#! Why
fps_grid_pc = pca_batch_transform(jt.map(
    lambda fps: subdict(fps, lohi(hps_common.train.pert.std)),
    fps_grid_pre_pc,
    is_leaf=LDict.is_of('train__pert__std'),
))

all_readout_weights_pc = jt.map(
    lambda weights: subdict(weights, lohi(hps_common.train.pert.std)),
    all_readout_weights_pc,
    is_leaf=LDict.is_of('train__pert__std'),
)

```

```python
def plot_fp_pcs_by_context(
    fp_pcs_by_context: LDict, 
    readout_weights_pc: Optional[Array] = None,
):
    fig = go.Figure(
        layout=dict(
            width=800,
            height=800,
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3',
            ),
            legend=dict(
                title='Context input',
                itemsizing='constant',
                y=0.85,
            ),
        )
    )
    
    for context, fps_pc in fp_pcs_by_context.items():
        fig = plot_fp_pcs(
            fps_pc, 
            fig=fig, 
            label=context,
            colors=colors["context_input"].dark[context],
        )
    
    if readout_weights_pc is not None:
        fig.update_layout(
            legend2=dict(
                title='Readout components',
                itemsizing='constant',
                y=0.45,
            ),
        )
        
        # Take the spatial average of the FPs for the lowest context input, at which 
        # to place the readout vectors in the visualization.
        mean_base_fp_pc = jnp.mean(fp_pcs_by_context[min(hps_common.context_input)], axis=1)
        
        traces = []
        k = 0.25
        for j in range(readout_weights_pc.shape[-2]):
            start = mean_base_fp_pc
            end = mean_base_fp_pc + k * readout_weights_pc[..., j, :]
            
            # Interleave start and end points with None for multiple disconnected lines
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
                    name=j,
                    legend="legend2",
                )
            )

        fig.add_traces(traces)

    return fig
```

```python
figs_fps_grid = jt.map(
    lambda fp_pcs_by_context, readout_weights_pc: plot_fp_pcs_by_context(
        fp_pcs_by_context, readout_weights_pc
    ),
    fps_grid_pc,
    all_readout_weights_pc,
    is_leaf=LDict.is_of("context_input")
)

# figs_fps_grid['std'][0.4]
```

```python
figs_fps_grid[0]
```

## Eigendecomposition of the steady-state Jacobians

Now that we have fixed points (network states) for the steady states, we can linearize the network around these points.

:::{note}
Ideally, for each model we want a function that takes a context input and returns a square Jacobian matrix (i.e. from hidden states to hidden states). However, note that we had to compute the fixed points *given* the context input. So for now we will simply calculate a set of Jacobians for the context inputs we *did* evaluate. 

In the future, we may be able to get those functions-of-context by wrapping the fixed point calculation up with the Jacobian calculation, however that might be kind of messy given that the fixed point calculation involves some filtering of candidates. 
:::


### Define functions

```python
def get_jac_func(position, context, func):
    return jax.jacobian(get_ss_rnn_func_at_context(position, context, func))
    
def get_jacobian(position, context, fp, func):
    return get_jac_func(position, context, func)(fp)
```

```python
@eqx.filter_jit
def get_all_jacobians(positions, all_fps, all_funcs):
    get_jac = eqx.filter_vmap(  # Over replicates
        get_jacobian, 
        in_axes=(None, None, 0, 0),
    )
    
    if isinstance(positions, Array) and len(positions.shape) == 2:
        get_jac = eqx.filter_vmap(  # Over positions
            get_jac,
            in_axes=(0, None, 1, None),  
        )
    
    def _get_jac_by_context(func, fps_by_context):
        return LDict.of('context_input')({
            context_input: get_jac(  
                positions, context_input, fps, first(func, is_leaf=is_module),
            )
            for context_input, fps in fps_by_context.items()
        })
    
    jacobians = jt.map(
        _get_jac_by_context,
        all_funcs, all_fps,
        is_leaf=LDict.is_of('context_input'),
    )
    
    jacobians_stacked = jt.map(
        lambda d: jtree.stack(list(d.values())),
        jacobians,
        is_leaf=LDict.is_of("context_input"),
    )
    
    return jacobians_stacked
```

### Compute all Jacobians

```python
goals_pos.shape
```

```python
best_replicate_only = True
origin_only = False

goals_pos_for_jac = goals_pos

if best_replicate_only:
    fps_grid_for_jac = get_best_replicate(fps_grid, replicate_info=replicate_info, axis=0, keep_axis=True)
    rnn_funcs_for_jac = get_best_replicate(rnn_funcs, replicate_info=replicate_info, axis=0, keep_axis=True)
else: 
    fps_grid_for_jac = fps_grid
    rnn_funcs_for_jac = rnn_funcs
    
if origin_only:
    idx = jnp.array([origin_idx])
    fps_grid_for_jac = jt.map(lambda x: x[:, idx], fps_grid_for_jac)
    goals_pos_for_jac = goals_pos[idx]

#! TODO: Best replicate only! (Or, do it before plotting)
jacobians_grid = get_all_jacobians(
    goals_pos_for_jac, fps_grid_for_jac, rnn_funcs_for_jac,  
)
```

### Eigendecomposition of Jacobians

How does the context input affect the stability the local linearization at steady-state fixed points? To see this, we can compare the eigenspectra of the Jacobians.

It's easier to just do this on the CPU since we need non-symmetric decomposition and the matrix is only 100x100 or so.

```python
eig_cpu = jax.jit(
    lambda *args, **kwargs: tuple(jax.lax.linalg.eig(*args, **kwargs)), 
    device=jax.devices('cpu')[0],
)
```

```python
eigvals_grid, eigvecs_l_grid, eigvecs_r_grid = tree_unzip(jt.map(eig_cpu, jacobians_grid))
```

:::{note}
Where are the eigenvalues non-NaN across all context inputs?

```python
eigvals_valid = jt.map(
    lambda eigvals: jnp.all(~jnp.isnan(
        jnp.nanmean(eigvals, axis=-1)  
    ), axis=0), 
    eigvals_grid,
)
```

Sanity check: this should look the same (nearly?) as `satisfactory_replicates`:

```python
eqx.tree_pprint(tree_stack([eigvals_valid, satisfactory_replicates_grid]), short_arrays=False)
```

```python
eigvals_valid_idxs = jt.map(
    lambda x: jnp.where(x), 
    eigvals_valid, 
)
```

In the origin-steady-state case, we could exclude entire replicates based on whether the eigenvalues are NaN for at least one context input. 

**However, for the grid case, for a single replicate, some steady-state points may have all valid eigenvalues while others do not, for the same context input. Thus while some replicates are clearly worse than others (e.g. for *every* steady-state point, at least one of the context inputs has NaN eigenvalues, and thus an entire row of `eigvals_valid` is `False`), it may not be the case that any row of `eigvals_valid` is all `True`.** This is why I've put this material in a callout; the indexing by `eigvals_valid_idxs` does not work in the grid case. Instead, for finding distributions I will lump over all replicates, and assuming the variation between replicates is smaller than the variation between context inputs, this should work well.

```python
# eigvals_final = jt.map(
#     lambda eigvals, idxs: jt.map(lambda vals: vals[:, idxs], eigvals),
#     eigvals, eigvals_valid_idxs,
#     is_leaf=is_type(ContextInputDict),
# )
```

```python
colors["context_input"].dark
```

:::

```python
from jaxtyping import ArrayLike


def find_indices(arr, values: Array | Sequence[ArrayLike]):
    """Find the indices of `values` in `arr`."""
    def find_single_value(value):
        return jnp.where(arr == value, size=1)
    
    # Vectorize this function across all values
    return jax.vmap(find_single_value)(jnp.array(values))


```

```python
# Optionally, only visualize a single replicate rather than lumping
i_replicate = None

# Optionally, only plot a subset of context inputs
contexts_plot = [-3, 0, 3]
context_idxs_plot = find_indices(jnp.array(hps_common.context_input), contexts_plot)[0].ravel()
context_colors = subdict(colors["context_input"].dark, contexts_plot)
```

```python
context_idxs_plot
```

```python
if i_replicate is not None:
    take_func_ = lambda arr: jnp.take(arr, i_replicate, axis=-2)
    col_names =['context', 'pos', 'eigenvalue']
else:
    take_func_ = lambda arr: arr 
    col_names = ['context', 'pos', 'replicate', 'eigenvalue'] 
    
if context_idxs_plot is not None:
    def take_func(arr):
        arr = take_func_(arr)
        return jnp.take(arr, context_idxs_plot, axis=0)
else:
    take_func = take_func_


eigval_dfs = jt.map(
    lambda arr: create_arr_df(
        take_func(arr),
        col_names=col_names,
    ).astype({'context': 'str', 'replicate': 'str'}),
    eigvals_grid,
)
```

```python
eigval_figs = jt.map(
    lambda eigvals: plot_eigvals_df(
        eigvals,
        marginals='box',
        color='context',
        color_discrete_sequence=list(context_colors.values()),
        trace_kws=dict(marker_size=2),
        scatter_kws=dict(opacity=1),
        layout_kws=dict(
            legend_title='Context input', 
            legend_itemsizing='constant',
            xaxis_title='Re',
            yaxis_title='Im',
        ),
    ),
    eigval_dfs,
)

# Reverse all the traces, since this improves the visualization in this particular case
# jt.map(
#     lambda fig: setattr(fig, 'data', fig.data[::-1]),
#     eigval_figs,
#     is_leaf=is_type(go.Figure),
# )

def _update_trace_name(trace):
    if trace.name is not None and trace.name != 'grid':
        return trace.update(name=contexts_plot[int(trace.name)])
    else:
        return trace
        

# Label the traces/legend with actual context inputs, rather than indices 
jt.map(
    lambda fig: fig.for_each_trace(_update_trace_name),
    eigval_figs,
    is_leaf=is_type(go.Figure),
);
```

```python
from rnns_learn_robust_motor_policies.plot import set_axes_bounds_equal

trace_selector = lambda trace: trace.type.startswith('scatter') and trace.name != 'grid'

figs = set_axes_bounds_equal(eigval_figs, trace_selector=trace_selector, padding_factor=0.02)
```

```python
from rnns_learn_robust_motor_policies.config.config import PATHS
from rnns_learn_robust_motor_policies.database import savefig


plot_id = 'steady_state_jacobian_eigvals/by_context/grid'
print(plot_id)

for path, fig in tqdm(figs_flatten_with_paths(figs)):    
    label = '_'.join([str(p.key) for p in path])
    print(label)
    fig.show()
    savefig(fig, label, PATHS.figures_dump, ['svg', 'webp'])
```

```python

```

### Plot eigenvalues by replicate

```python
import plotly.express as px
```

```python
eigval_figs = jt.map(
    lambda eigvals: plot_eigvals_df(
        eigvals,
        marginals='box',
        color='replicate',
        color_discrete_sequence=px.colors.qualitative.G10,
        trace_kws=dict(marker_size=3),
        layout_kws=dict(
            legend_title='Replicate', 
            legend_itemsizing='constant',
            xaxis_title='Re',
            yaxis_title='Im',
        ),
    ),
    eigval_dfs,
)
```

```python
plot_id = 'steady_state_jacobian_eigvals/by_replicate/grid'
print(plot_id)

for path, fig in tqdm(figs_flatten_with_paths(eigval_figs)):
    fig_params = dict(
        training_method=path[0].key,
        disturbance_type_train=disturbance_type_load,
        disturbance_train_std=path[1].key,
        n=model_info_0.hidden_size * len(context_inputs) * eval_grid_n ** 2,
    )
    
    add_evaluation_figure(
        db_session, 
        eval_info, 
        fig, 
        plot_id, 
        model_records=model_info[path[0].key][path[1].key],
        **fig_params,
    )
    
    print(path[0].key, path[1].key)
    fig.show()
```

### Plot eigenvalues by position

```python
eigval_figs = jt.map(
    lambda eigvals: plot_eigvals_df(
        eigvals,
        marginals=None,  # Too many positions for this to work well
        color='pos',
        trace_kws=dict(marker_size=3),
        layout_kws=dict(
            legend_title='Replicate', 
            legend_itemsizing='constant',
            xaxis_title='Re',
            yaxis_title='Im',
        ),
    ),
    eigval_dfs,
)
```

```python
plot_id = 'steady_state_jacobian_eigvals/by_position/grid'
print(plot_id)

for path, fig in tqdm(figs_flatten_with_paths(eigval_figs)):
    fig_params = dict(
        training_method=path[0].key,
        disturbance_type_train=disturbance_type_load,
        disturbance_train_std=path[1].key,
        n=model_info_0.hidden_size * len(context_inputs) * model_info_0.n_replicates,
    )
    
    add_evaluation_figure(
        db_session, 
        eval_info, 
        fig, 
        plot_id, 
        model_records=model_info[path[0].key][path[1].key],
        **fig_params,
    )
    
    print(path[0].key, path[1].key)
    fig.show()
```

### Quantify variation of eigenvalues across context inputs, versus replicates, versus grid positions

```python
eigval_stds = {
    var_label: jt.map(
        # TODO: Could take the nanmean over axis=-1 (i.e. eigenvalue idx) here
        lambda eigvals: jnp.nanstd(eigvals, axis=axis),
        eigvals_grid,
    )
    for axis, var_label in enumerate(['context', 'position', 'replicate'])
}
```

Aggregate over other dimensions. I'm not sure of the best statistic here, but I think calculating both the mean and the max should give a good idea of the variation.

```python
eigvals_std_stats = jt.map(
    lambda stds: (jnp.nanmean(stds).item(), jnp.nanmax(stds).item()),
    eigval_stds,
)

pp(eigvals_std_stats)
```

**So the mean stds are modestly higher for context inputs (and they increase with train std. for the "std" method) but they are still significant for replicate and position. I think that this might be the case because eigenvalues may not maintain their ordering (if there is even a sensible way to define that) such that these stds include information about the variation *across* eigenvalues, i.e. their spread.** The grid positions and replicates are lumped together in the eigenvalue plots, in which the dependence on context input is clear, which suggests that the variation in these other variables is happening on a smaller scale. One feature that seems to vary between replicates is the location of the "wings".

Note that this calculation is taking the absolute value of the complex eigenvalues before computation. If we perform the same calculation on the real and imaginary parts separately, then most of the variation, particularly in the context input, appears to be in the real part. 

## Feedback perturbation at steady-state

:::{note}
We don't need to perturb the task here necessarily -- not even to get the candidate FPs. We can probably use the steady-state FPs as the candidates (perhaps augmented with some jitter to get a batch) and simply ramp the relevant network input and repeat the FP calculation. We can do this on a smaller grid than before, since we want to make sure that there isn't huge variation between FPs but also this variation isn't our primary concern here. 
:::

Here, the "init-goal" FPs correspond to a steady-state network input where one of the feedback channels has been perturbed. 

In particular, we could ramp up the perturbation from zero, and see how the FP changes, relative to the steady-state one.

To start with we can look at 1) just the origin steady-state, and 2) perturbations just to velocity feedback, with a series of fixed and increasing magnitudes, and in a spread of directions around the origin.


```python
perturbation_amplitudes = jnp.array([0.2, 0.3, 0.4])

n_perturbation_directions = 1
```

```python
pert_amp_colors, pert_amp_colors_dark = get_colors_dicts(
    perturbation_amplitudes.tolist(), COLORSCALES['disturbance_amplitudes'],
)
```

```python
from jaxtyping import Float, Array

def get_network_input_with_context_and_fb_pert(pos, context, fb_pert, rnn_cell):
    input_star = jnp.zeros((rnn_cell.input_size,))
    # Set target and feedback inputs to the same position
    input_star = input_star.at[1:3].set(pos) 
    input_star = input_star.at[5:7].set(pos)
    input_star = input_star.at[5:9].add(fb_pert)
    return input_star.at[0].set(context)
    
    
def get_rnn_func(pos, context, fb_pert, rnn_cell):
    input_star = get_network_input_with_context_and_fb_pert(pos, context, fb_pert, rnn_cell)
    
    def rnn_func(h):
        return rnn_cell(input_star, h, key=key_eval)
    
    return rnn_func


def augment_state(state, n_samples, scale, *, key):
    return state + scale * jax.random.normal(key, (n_samples,) + state.shape)
    
n_augment = 50
augment_scale = 1e-2
    
@partial(eqx.filter_vmap, in_axes=(None, None, 0, 0, None, 0)) # Over replicates
@partial(eqx.filter_vmap, in_axes=(0, None, None, 0, None, 0)) # Over grid positions
@partial(eqx.filter_vmap, in_axes=(None, 0, None, None, None, None)) # Over perturbation directions
@partial(eqx.filter_vmap, in_axes=(None, 0, None, None, None, None)) # Over perturbation amplitudes
def get_rnn_fp_with_fb_pert(pos, fb_pert, rnn_cell, candidate_state, context, key):
    """Find the fixed point """       
    fps = fpf_func(
        get_rnn_func(pos, context, fb_pert, rnn_cell), 
        augment_state(candidate_state, n_augment, augment_scale, key=key),
        # candidate_state,
        fp_tol,
    )
    return fps
    
    
def get_centerout_vel_pert_arrs(n_directions, amplitude):
    """Get feedback vectors with zero position and evenly-spread center-out velocity."""
    return jnp.pad(
        centreout_endpoints(jnp.zeros(2), n_directions, amplitude)[1],
        ((0, 0), (2, 0)),
    )
    
    
@eqx.filter_jit
def get_fps_at_positions_with_fb_perts(
    positions: Float[Array, "n space=2"], 
    fb_perts: Float[Array, "amplitude direction 4"],
    all_models: PyTree[eqx.Module, 'T'], 
    all_ss_fps: PyTree[ContextInputDict, 'T'],
    context_inputs: Sequence[float],
    stride_candidates: int = 1,
):
    """For each of one or more workspace positions, find the respective fixed point of the RNN
    whose inputs indicate it's reached a goal at that position.
    
    Repeat over a range of fixed values of the RNN's context input.
    
    Exclude any replicate for which no fixed point meeting the acceptance criteria was found, 
    for at least one context input.
    """        
    # candidates = jt.map(
    #     lambda fps: jnp.reshape(
    #         fps, 
    #         (model_info_0.n_replicates, -1, model_info_0.hidden_size),
    #     )[:, ::stride_candidates],
    #     all_ss_fps,
    #     is_leaf=is_module,
    # )

    keys = jr.split(key_eval, (model_info_0.n_replicates, len(positions)))

    return tree_map_tqdm(
        lambda models, candidates_by_context: ContextInputDict({
            context_input: get_rnn_fp_with_fb_pert(
                positions,
                fb_perts,
                models.step.net.hidden,
                candidates_by_context[context_input],
                context_input,
                keys,
            )
            for context_input in context_inputs
        }),
        all_models, all_ss_fps,
        is_leaf=is_module,
    )
```

```python
fb_perts = eqx.filter_vmap(get_centerout_vel_pert_arrs, in_axes=(None, 0))(
    n_perturbation_directions, perturbation_amplitudes
)
```

```python
args = (
    # positions,
    goals_pos, 
    fb_perts, 
    all_models, 
    # jt.map(lambda arr: jnp.expand_dims(arr, 1), fps_origin),
    fps_grid,
    context_inputs,
)
```

```python
fp_results_shape = eqx.filter_eval_shape(
    get_fps_at_positions_with_fb_perts,
    *args,
)

print(f"Estimate {tree_struct_bytes(fp_results_shape) / 1E6} MB of memory needed for all FPs.")
```

```python
# leaf shape: (replicates, grid points, perturbation amplitudes, perturbation directions, fp candidates (=1), hidden size)
fp_results = get_fps_at_positions_with_fb_perts(*args)

# leaf shape: (replicates, grid points, perturbation amplitudes, perturbation directions, hidden size)
fps_pert, all_top_fps_pert, n_fps_meeting_criteria_pert, satisfactory_replicates_pert = process_fps(fp_results)
del fp_results
```

As earlier, assume that if there are multiple FPs returned, they are actually the same FP due to slight variations in convergence; inside `process_fps` we average over them.

:::{note}
Note that currently, definitely only one FP will be returned because we're only passing a single candidate (the steady-state FP for the respective condition) to the optimization. See the `[None, :]` expansion in the FP-finding function above, which is necessary since we vmapped over both the replicate *and* the grid position. In the future we might instead pass multiple candidates, e.g. by jittering the steady-state FP, or by not vmapping over the grid position for the candidates. 
:::

### Analysis rationale

Note that while these are not steady-state FPs with respect to the entire system, they are still FPs with respect to the network with its inputs fixed.

As we scale up the perturbation to its inputs, we expect the FP to move, and its eigenspectrum perhaps to change. For very small perturbations, the network may need to move only a very small distance in state space, and the linear dynamics may be almost identical. For larger perturbations, the network will probably not reach the FP before its inputs change again due to the closed-loop feedback. Thus the linearization may not describe the network's closed-loop behaviour except perhaps in the limit of it being very far from its goal?

On the other hand, this analysis may correspond better to the open-loop response, in the case where we introduce the perturbation but do not allow the network to be influenced by the change in feedback due to the movement of the point mass, while the perturbation is active.

### Analysis of FP structure

How do the perturbations alter the position of FPs? How does this depend on context input? 

```python
fps_pert_pre_pc = jt.map(
    lambda fps: take_non_nan(fps, axis=1),
    exclude_nan_replicates(fps_pert),
)

aa = jt.map(
    lambda d: ContextInputDict({k: v for k, v in d.items() }),
    fps_pert_pre_pc,
    is_leaf=is_type(ContextInputDict),
)

# Leaf shape: (included replicates, grid points, perturbation amplitudes, perturbation directions, hidden size)
fps_pert_pc = jt.map(
    lambda fps: batch_reshape(pca.transform)(fps),
    aa,
)
```

```python
fps_pert_pc_ = jt.map(
    lambda fps: fps[..., 0, 0, :],
    fps_pert_pc,
)

figs_fps_grid = jt.map(
    lambda fp_pcs_by_context, readout_weights_pc: plot_fp_pcs_by_context(
        fp_pcs_by_context, readout_weights_pc
    ),
    fps_pert_pc_,
    all_readout_weights_pc,
    is_leaf=is_type(ContextInputDict),
)

figs_fps_grid['std'][1.2]
```

#### Supplementary

Is this structure consistent across grid points? Across replicates?

### Eigenspectra of Jacobians

How do the perturbations alter the eigenspectra of the Jacobians? How does this depend on context input?

**TODO**: Generalize this with the version without perturbations from earlier. Probably, just partial out the `fb_pert` and then use `squeeze` to eliminate the superfluous dimension in the result.

```python
def get_jac_func_with_pert(position, context, fb_pert, func):
    return jax.jacobian(get_rnn_func(position, context, fb_pert, func))

@partial(eqx.filter_vmap, in_axes=(None, None, None, 0, 0)) # Over replicates
@partial(eqx.filter_vmap, in_axes=(None, 0, None, 0, None)) # Over grid positions
@partial(eqx.filter_vmap, in_axes=(1, None, None, 1, None)) # Over perturbation direction
@partial(eqx.filter_vmap, in_axes=(0, None, None, 0, None)) # Over perturbation amplitude
def get_jacobian(fb_pert, position, context, fp, func):
    return get_jac_func_with_pert(position, context, fb_pert, func)(fp)
```

```python
@eqx.filter_jit
def get_all_jacobians(fb_perts, positions, all_fps, all_models):    
    jacobians = jt.map(
        lambda models, fps_by_context: ContextInputDict({
            context_input: get_jacobian(  
                fb_perts, positions, context_input, fps, models.step.net.hidden,
            )
            for context_input, fps in fps_by_context.items()
        }),
        all_models, all_fps,
        is_leaf=is_module,
    )

    jacobians_stacked = stack_by_subtree_type(jacobians, ContextInputDict)
    
    return jacobians_stacked
```

```python
jacobians_pert = get_all_jacobians(
    fb_perts, goals_pos, fps_pert, all_models,
)
# leaf shape: (context, replicate, position, perturbation direction, perturbation_amplitude, hidden size, hidden size)
```

```python
# If we use `eig_cpu` from earlier, then we get 5-10 GB of returned eigvecs, which
# is much more memory than used by the rest of this notebook. Thus until we need them,
# we won't compute them.

get_eigvals_cpu = jax.jit(
    lambda *args, **kwargs: tuple(jax.lax.linalg.eig(
        *args, 
        compute_left_eigenvectors=False, 
        compute_right_eigenvectors=False, 
        **kwargs,
    )), 
    device=jax.devices('cpu')[0],
)

eigvals_pert = tree_unzip(jt.map(get_eigvals_cpu, jacobians_pert))[0]
# leaf shape: (context, replicate, position, perturbation direction, perturbation amplitude, hidden size)
```

#### Compare eigenspectra by perturbation amplitude

```python
col_names = ['context', 'replicate', 'pos', 'pert.dir.', 'pert.amp.', 'eigenvalue']

eigval_pert_dfs = jt.map(
    lambda arr: create_arr_df(
        arr,
        col_names=col_names,
    ).astype({'context': 'str', 'replicate': 'str', 'pert.amp.': 'str', 'pert.dir.': 'str'}),
    eigvals_pert,
)
```

```python
condition_func = lambda df: df[
    (df['pert.dir.'] == '0') 
    & (df['pos'] == origin_idx)
    & (df['replicate'] == '1')
    & (df['context'] == '4')
]
```

```python
from rnns_learn_robust_motor_policies.misc import round_to_list

eigval_figs = jt.map(
    lambda eigval_df: plot_eigvals_df(
        condition_func(eigval_df),
        marginals='box',
        color='pert.amp.',
        color_discrete_sequence=list(pert_amp_colors_dark.values()),
        # trace_kws=dict(marker_size=3),
        layout_kws=dict(
            legend_title='Pert. amplitude', 
            legend_itemsizing='constant',
            xaxis_title='Re',
            yaxis_title='Im',
        ),
    ),
    eigval_pert_dfs,
)

# Reverse all the traces, since this improves the visualization in this particular case
# jt.map(
#     lambda fig: setattr(fig, 'data', fig.data[::-1]),
#     eigval_figs,
#     is_leaf=is_type(go.Figure),
# )

ff = round_to_list(perturbation_amplitudes)

# Label the traces/legend with actual variable values, and not indices
jt.map(
    lambda fig: fig.for_each_trace(
        lambda t: t.update(name=ff[int(t.name)]) 
        if t.name is not None else t
    ),
    eigval_figs,
    is_leaf=is_type(go.Figure),
);
```
