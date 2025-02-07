
import logging
import logging.handlers as loghandlers
import os
from pathlib import Path

import plotly.io as pio
import yaml

from rnns_learn_robust_motor_policies.config import load_named_config
# from rnns_learn_robust_motor_policies.training import train_and_save_models


LOG_LEVEL = "DEBUG"

prng_config = load_named_config('prng')
PROJECT_SEED: int = prng_config['seed']

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
BEST_MODEL_FILE_LABEL = f"{MODEL_FILE_LABEL}__best_params"
HYPERPARAMS_FILE_LABEL = "hyperparameters"
REPLICATE_INFO_FILE_LABEL = "replicate_info"
TRAIN_HISTORY_FILE_LABEL = "train_history"


# Set the default Plotly theme
pio.templates.default = 'simple_white'


# Logging configuration
logger = logging.getLogger(__package__)
logger.setLevel(LOG_LEVEL)
file_handler = loghandlers.RotatingFileHandler(
    f"{__package__}.log",
    maxBytes=1_000_000,
    backupCount=1,
)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s,%(lineno)d: %(message)s",
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.captureWarnings(True)

logger.info("Logger configured.")

