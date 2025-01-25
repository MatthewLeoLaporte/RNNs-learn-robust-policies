
import os
from pathlib import Path

import plotly.io as pio
import yaml

from rnns_learn_robust_motor_policies.config import load_named_config
from rnns_learn_robust_motor_policies.misc import load_yaml
# from rnns_learn_robust_motor_policies.training import train_and_save_models

PROJECT_SEED = 5566


# Directory configuration
paths = load_named_config('paths')
(DB_DIR, MODELS_DIR, FIGS_BASE_DIR, QUARTO_OUT_DIR) = [
    path for path in [
        Path(paths[label])
        for label in ('db_dir', 'models_dir', 'figs_dir', 'quarto_outputs')
    ]
]
for d in (DB_DIR, MODELS_DIR, FIGS_BASE_DIR, QUARTO_OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Labels for constructing and parsing file names
MODEL_FILE_LABEL = "trained_models"
BEST_MODEL_FILE_LABEL = f"{MODEL_FILE_LABEL}_best_params"
HYPERPARAMS_FILE_LABEL = "hyperparameters"
REPLICATE_INFO_FILE_LABEL = "replicate_info"
TRAIN_HISTORY_FILE_LABEL = "train_history"


# Set the default Plotly theme
pio.templates.default = 'simple_white'

