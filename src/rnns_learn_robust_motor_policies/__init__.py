from pathlib import Path

import plotly.io as pio
import yaml

from rnns_learn_robust_motor_policies.misc import load_yaml


PROJECT_SEED = 5566


# Directory configuration
REPO_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = REPO_ROOT / 'config'
paths = load_yaml(CONFIG_DIR / 'paths.yaml')['paths']
(MODELS_DIR, FIGS_BASE_DIR) = [
    Path(paths[label])
    for label in ('models_dir', 'figs_base_dir')
]


# Labels for constructing and parsing file names
MODEL_FILE_LABEL = "trained_models"
BEST_MODEL_FILE_LABEL = f"{MODEL_FILE_LABEL}_best_params"
HYPERPARAMS_FILE_LABEL = "hyperparameters"
REPLICATE_INFO_FILE_LABEL = "replicate_info"


# Set the default Plotly theme
pio.templates.default = 'simple_white'