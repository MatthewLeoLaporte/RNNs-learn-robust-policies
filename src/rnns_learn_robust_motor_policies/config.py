from pathlib import Path 
import yaml

from rnns_learn_robust_motor_policies import CONFIG_DIR


def load_default_config(nb_id: str) -> dict:
    """Load config from file or use defaults"""
    return load_config(CONFIG_DIR / f"{nb_id}.yml")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    return config