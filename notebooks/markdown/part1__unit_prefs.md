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
ANALYSIS_NAME = "part1.unit_prefs"
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

```python
hps = common_data['hps_common']
```

```python
prefs_key = 'Prefs: control forces'

prefs = jt.map(
    lambda d: d["decel"],
    all_results[prefs_key]["full"], 
    is_leaf=LDict.is_of("epoch")
)

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

What is the distribution of preferred force angles in the baseline condition (no training or testing
perturbations)?

```python
_ = circular_hist(pref_angles[0][0])
```

How does this compare to the distribution of preferred *instantaneous* force directions implied by
the readout?

```python
train_pert_std = 0
best_replicate = common_data['replicate_info'][train_pert_std]['best_replicates']['best_total_loss']
readout_weights = data.models["full"][0][train_pert_std].step.net.readout.weight[best_replicate]
pref_angles_readout = vectors_to_2d_angles(readout_weights.T)

_ = circular_hist(pref_angles_readout, mean=True)
```

Does it make sense to divide by these counts? i.e. express the bin counts as proportions of the
value implied by the readout.

```python
_ = circular_hist(pref_angles[0][0], normalize_by_other=pref_angles_readout)
```

How does the preferred force angle change when a perturbation is added?

```python
common_data.keys()
```

```python
for train_pert_std in hps.train.pert.std:
    pref_angle_change = angle_between_vectors(prefs[0][train_pert_std], prefs[4][train_pert_std])
    # print(f"Perturbation std: {train_pert_std}")
    fig, ax = circular_hist(pref_angle_change, mean=True)
    ax.set_title(f"Train pert. std.: {train_pert_std}")
    fig.show()

```

```python

```
