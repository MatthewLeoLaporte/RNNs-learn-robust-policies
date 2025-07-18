
import logging
import logging.handlers as loghandlers
import os
from pathlib import Path

import plotly.io as pio

from rlrmp.config import (
    CONFIG_DIR_ENV_VAR_NAME,
    PLOTLY_CONFIG,
    LOGGING_CONFIG,
)


# Set the default Plotly theme for the project
pio.templates.default = PLOTLY_CONFIG.templates.default


# Logging configuration
logger = logging.getLogger(__package__)
logger.setLevel(LOGGING_CONFIG.level)
file_handler = loghandlers.RotatingFileHandler(
    f"{__package__}.log",
    maxBytes=1_000_000,
    backupCount=1,
)
formatter = logging.Formatter(LOGGING_CONFIG.format_str)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logging.captureWarnings(True)
logger.info("Logger configured.")

