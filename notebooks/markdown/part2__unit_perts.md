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

# Network unit perturbations




```python
ANALYSIS_NAME = "part2.unit_perts"
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
import warnings

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
from feedbax.plot import circular_hist
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
from rnns_learn_robust_motor_policies.analysis.state_utils import get_best_replicate, exclude_bad_replicates, angle_between_vectors
from rnns_learn_robust_motor_policies.colors import (
    COLORSCALES,
    MEAN_LIGHTEN_FACTOR, COMMON_COLOR_SPECS,
)
from rnns_learn_robust_motor_policies.config import PRNG_CONFIG, PATHS
from rnns_learn_robust_motor_policies.database import add_evaluation_figure, savefig
from rnns_learn_robust_motor_policies.misc import create_arr_df, take_non_nan, lohi, vectors_to_2d_angles
from rnns_learn_robust_motor_policies.plot import (
    plot_eigvals_df,
    plot_fp_pcs,
    set_axes_bounds_equal,
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

```python
warnings.filterwarnings('ignore')
```

```python
import plotly.io as pio

pio.templates.default = "simple_white"
```

```python
key = jr.PRNGKey(PRNG_CONFIG.seed)
_, _, key_eval = jr.split(key, 3)

data, common_data, _, all_results, _ = run_analysis_module(ANALYSIS_NAME, key=key)
```

Keep the model parameters and states only for the best replicate, for the remaining analyses. 

```python
all_states = get_best_replicate(
    data.states['full'],
    replicate_info=common_data['replicate_info'],
    axis=3,
)

all_models = get_best_replicate(
    data.models["full"],
    replicate_info=common_data['replicate_info'],
    axis=0,
)

hps_common = common_data['hps_common']
```

Get the 

```python
prefs_readout = LDict.of('train__pert__std')({
    tps: all_models[0][0][tps].step.net.readout.weight.T
    for tps in hps_common.train.pert.std
})

pref_angles_readout = jt.map(vectors_to_2d_angles, prefs_readout)
```

## Examine unit preferences

During the acceleration period

```python
prefs = jt.map(
    lambda d: d["accel"],
    all_results['unit_prefs'], 
    is_leaf=LDict.is_of("epoch")
)['full']

pref_angles = jt.map(vectors_to_2d_angles, prefs)
```

```python
tree = pref_angles 

while True:
    try:
        print(tree.label, '\t', list(tree.keys()))
        tree = next(iter(tree.values()))
    except AttributeError:
        break
```

### Exploratory comparison of preferred angle change between conditions

```python
REF_LINE_RADIUS = 0.25

stim = jnp.array([True, True], dtype=int)
stim_unit_idx = jnp.array([1, 1], dtype=int)

context_input = [-2, 2]
pert__amp = [0.0, 0.0]
train__pert__std = [1.5, 1.5]

prefs_to_compare = [
    prefs[c][a][s][i, j]
    for c, a, s, i, j in zip(context_input, pert__amp, train__pert__std, stim, stim_unit_idx)
]

pref_angle_change = angle_between_vectors(*prefs_to_compare)

fig, ax = circular_hist(pref_angle_change, mean=True)

for i, tps in enumerate(train__pert__std):
    for j, sui in enumerate(stim_unit_idx):
        stim_unit_pa = pref_angles_readout[tps][sui]
        ax.plot([stim_unit_pa] * 2, [0, REF_LINE_RADIUS], color=f"C{i}")
```

### Exploratory measures of response change between conditions


1. What is the max. deviation from steady-state position for each stim, and in what direction? 
2. What about the final deviation?

```python
pos = jt.map(
    lambda states: states.mechanics.effector.pos,
    all_states,
    is_leaf=is_module,
)

print(tree_level_labels(pos))
print(jt.leaves(pos)[0].shape)
```

```python
LAST_N_TIME_STEPS_MEAN = 5

# all the trials are at the origin (0, 0); just take norm
deviations = jt.map(partial(jnp.linalg.norm, axis=-1), pos)

deviations_final, deviations_final_directions = jt.map(
    lambda arr: jnp.mean(arr[..., -LAST_N_TIME_STEPS_MEAN:], axis=-1), 
    (deviations, pos),
)

# time step of max deviation
deviations_max_idxs = jt.map(partial(jnp.argmax, axis=-1), deviations)

# TODO: jnp.expand_dims(max_idx, axis=(-1, -2, ...))
# direction of max deviation
deviations_max = jt.map(
    lambda pos, max_idx: jnp.take_along_axis(pos, max_idx[..., None], axis=-1),
    deviations,
    deviations_max_idxs,
)

# positions = directions since all trials are at the origin
deviations_max_directions = jt.map(
    lambda pos, max_idx: jnp.take_along_axis(pos, max_idx[..., None, None], axis=-2),
    pos,
    deviations_max_idxs,
)
```

1. How aligned are the max. deviation directions with the instantaneous/observational PD of the stim
   unit?

```python
# aggregated unit-wise alignment between max deviation direction and readout direction
max_dev_vs_readout = jt.map(
    lambda devs_max_dirs_by_std: jt.map(
        angle_between_vectors,
        pref_angles_readout,
        devs_max_dirs_by_std,
    ),
    deviations_max_directions,
    is_leaf=LDict.is_of("train__pert__std"),
)

alignment_max_dev_vs_readout = jt.map(
    lambda arr: jnp.mean(jnp.abs(arr)).item(),
    max_dev_vs_readout,
)
```

```python
alignment_max_dev_vs_readout
```

2. How aligned are the max. and final deviations?

```python
# TODO
```

3. Do the deviations change in magnitude for different values of the context input? How
   does this interact with the curl field?

```python
# TODO
```

## Examine unit activities

```python
stim_unit_idx = 51
stim = int(True)
replicate_i = 2

activities = data.states['full'][0][0][0].net.hidden[stim, stim_unit_idx, :, replicate_i] 
print(activities.shape)
fbp.activity_sample_units(activities, key=jr.PRNGKey(1), unit_includes=[stim_unit_idx])
```

```python

```
