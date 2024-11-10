"""

Replacement for `render.py`. Here, `render_notebook` will load all trained models that the notebook is able to process
from the database of trained models. It also loads all combination rules (for the respective notebook) from `eval_rules.yml`,
and adds combinations to satisfy the rules.

It then renders the notebook for each combination, calling `.database.add_notebook_evaluation` each time, 
resulting in the evaluation being logged (along with all its figures) in the evaluations database.

Written with the help of Claude 3.5 Sonnet.
"""

from dataclasses import dataclass
from itertools import product
import jsonschema
from pathlib import Path
from sqlalchemy.orm import Session
import subprocess
from typing import Dict, List, Optional, Union, Any
import yaml

from .database import TrainedModel, add_notebook_evaluation, query_model_entries


@dataclass
class EvalRule:
    condition: Dict[str, Any]
    parameters: Dict[str, Union[Any, List[Any]]]


@dataclass
class NotebookEvalConfig:
    notebook_id: str
    train_notebook_id: str
    rules: List[EvalRule]
    default_parameters: Dict[str, Any]
    parameter_combinations: Optional[Dict[str, Union[Any, List[Any]]]] = None
    
    
def load_eval_configs(
    config_path: Path = Path("config/notebook_eval_rules.yaml"),
    schema_path: Path = Path("db/eval_rules.schema.yaml"),
) -> Dict[str, NotebookEvalConfig]:
    """Load and validate evaluation configurations from YAML file."""
    # Load schema
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    
    # Load and validate config
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
        
    try:
        jsonschema.validate(config_dict, schema)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Invalid configuration: {e.message}")
    
    # Convert to dataclasses
    configs = {}
    for notebook_id, notebook_config in config_dict["notebooks"].items():
        configs[notebook_id] = NotebookEvalConfig(
            notebook_id=notebook_id,
            train_notebook_id=notebook_config["train_notebook_id"],
            rules=[
                EvalRule(**rule_dict)
                for rule_dict in notebook_config.get("rules", [])
            ],
            default_parameters=notebook_config["default_parameters"]
        )
    
    return configs


EVAL_CONFIGS = load_eval_configs()


def expand_parameter_combinations(
    model_entry: Optional[TrainedModel],
    config: NotebookEvalConfig
) -> List[Dict[str, Any]]:
    """Generate all parameter combinations based on config and optional model."""
    # Start with base parameters
    if model_entry is None:
        # Training notebook case - use parameter_combinations
        if not config.parameter_combinations:
            base_combinations = [config.default_parameters]
        else:
            # Convert single values to lists for consistent processing
            param_lists = {
                k: [v] if not isinstance(v, list) else v
                for k, v in config.parameter_combinations.items()
            }
            
            base_combinations = []
            # Generate all combinations
            for values in product(*param_lists.values()):
                params = config.default_parameters.copy()
                params.update(dict(zip(param_lists.keys(), values)))
                base_combinations.append(params)
    else:
        # Analysis notebook case - use model attributes
        base_params = {}
        for param, default in config.default_parameters.items():
            if isinstance(default, str) and default.endswith('_load'):
                base_params[param] = getattr(model_entry, default)
            else:
                base_params[param] = default
        base_combinations = [base_params]

    # Apply rules to all base combinations
    final_combinations = []
    for base_params in base_combinations:
        matches_any_rule = False
        for rule in config.rules:
            # For training notebook, check conditions against base_params
            # For analysis notebook, check against model attributes
            conditions_met = all(
                (getattr(model_entry, k) if model_entry else base_params.get(k)) == v
                for k, v in rule.condition.items()
            )
            
            if conditions_met:
                matches_any_rule = True
                param_lists = {
                    k: [v] if not isinstance(v, list) else v
                    for k, v in rule.parameters.items()
                }
                for values in product(*param_lists.values()):
                    params = base_params.copy()
                    params.update(dict(zip(param_lists.keys(), values)))
                    final_combinations.append(params)
        
        # If no rules matched, keep the base parameters
        if not matches_any_rule:
            final_combinations.append(base_params)

    return final_combinations


def render_notebook(
    session: Session,
    notebook_id: str,
    output_base_dir: Path = Path("_output"),
    config: Optional[NotebookEvalConfig] = None
):
    """Render a notebook with all parameter combinations for matching models."""
    config = config or EVAL_CONFIGS[notebook_id]
    
    # Find all matching models (or None for training notebooks)
    if config.train_notebook_id is not None:
        models = query_model_entries(
            session, 
            filters={"notebook_id": config.train_notebook_id}
        )
    else:
        models = [None]
    
    for model in models:
        # Generate parameter combinations for this model/config
        param_combinations = expand_parameter_combinations(model, config)
        
        for params in param_combinations:
            # Create evaluation entry and get hash
            eval_entry = add_notebook_evaluation(
                session, 
                model.id.item() if model else None,
                notebook_id, 
                params,
                output_base_dir,
            )
            
            # Render notebook with parameters and eval hash
            cmd = ["quarto", "render", f"{notebook_id}_*.qmd", 
                  "--output-dir", eval_entry.output_dir]
            for k, v in params.items():
                if v is not None:
                    cmd.extend(["-P", f"{k}:{v}"])
            # Pass eval hash to notebook
            cmd.extend(["-P", f"eval_hash:{eval_entry.hash}"])
                    
            subprocess.run(cmd, check=True)