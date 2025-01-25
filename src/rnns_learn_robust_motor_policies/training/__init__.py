from .train import (
    make_delayed_cosine_schedule,
    concat_save_iterations,
    load_hps,
    fill_out_hps,
    does_model_record_exist,
    skip_already_trained,
    train_and_save_models,
    train_pair,
    train_setup,
)

from .part1_fixed import (
    setup_task_model_pair as setup_task_model_pair_p1,
)
from .part2_context import (
    setup_task_model_pair as setup_task_model_pair_p2
)

# Deserialisation depends on where/how the model was trained
TRAINPAIR_SETUP_FUNCS = {
    1: setup_task_model_pair_p1,
    2: setup_task_model_pair_p2,
}