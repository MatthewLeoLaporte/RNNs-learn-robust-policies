from typing import Any

from feedbax.intervene import (
    CurlField, 
    FixedField, 
)


INTERVENOR_LABEL = "DisturbanceField"
DISTURBANCE_CLASSES = {
    'curl': CurlField,
    'random': FixedField,
}


## Model parameters
MASS = 1.0


## Task parameters
# TODO: Maybe move to config/task.yml
N_STEPS = 100
WORKSPACE = ((-1., -1.),
             (1., 1.))
TASK_EVAL_PARAMS: dict[str, dict[str, Any]] = dict(
    full=dict(
        eval_grid_n=2,
        eval_n_directions=24,
        eval_reach_length=0.5,
    ),
    small=dict(
        eval_grid_n=1,
        eval_n_directions=7,
        eval_reach_length=0.5,
    ),
)