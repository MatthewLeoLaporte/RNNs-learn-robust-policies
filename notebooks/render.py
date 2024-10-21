import argparse
import os
import yaml
import subprocess
import logging
from typing import Generic, List, Dict, Any, TypeVar, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def apply_rules(combination: List[Any], parameter_names: List[str], rules: List[Dict[str, Any]]) -> List[List[Any]]:
    """Apply rules to a single combination and return new combinations."""
    new_combinations = [combination]
    for rule in rules:
        condition = list(rule['condition'].items())[0]
        actions = rule['action']
        
        if condition[0] in parameter_names:
            idx = parameter_names.index(condition[0])
            if combination[idx] == condition[1]:
                new_combo = combination.copy()
                for action_param, action_value in actions.items():
                    if action_param in parameter_names:
                        new_combo[parameter_names.index(action_param)] = action_value
                new_combinations.append(new_combo)
    
    return new_combinations

def apply_default_assignments(params: Dict[str, Any], default_assignments: List[Dict[str, str]]) -> Dict[str, Any]:
    """Apply default assignments to the parameters when they are null."""
    for assignment in default_assignments:
        for target_key, source_key in assignment.items():
            if params.get(target_key) is None:
                params[target_key] = params.get(source_key)
    return params

def expand_combinations(combinations: List[List[Any]], parameter_names: List[str], rules: List[Dict[str, Any]]) -> List[List[Any]]:
    """Expand all combinations by applying rules."""
    expanded = []
    for combo in combinations:
        expanded.extend(apply_rules(combo, parameter_names, rules))
    return [list(x) for x in set(tuple(x) for x in expanded)]  # Remove duplicates

def format_output_label(output_format: List[List[str]], params: Dict[str, Any]) -> str:
    """Format the output label based on the given format and parameters."""
    return '/'.join('_'.join(
        s.format(**params) for s in part
    ) for part in output_format)

def render_quarto(quarto_doc: str, output_dir: str, params: Dict[str, Any]):
    """Render a Quarto document with the given parameters."""
    cmd = ["quarto", "render", quarto_doc, "--output-dir", output_dir]
    for key, value in params.items():
        if value is not None:
            cmd.extend(["-P", f"{key}:{value}"])
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def process_quarto_doc(quarto_doc: str, combinations: List[List[Any]], parameter_names: List[str], output_format: List[List[str]], default_assignments: List[Dict[str, str]]):
    """Process a single Quarto document with all parameter combinations."""
    if not os.path.exists(quarto_doc):
        logger.warning(f"The file '{quarto_doc}' does not exist. Skipping.")
        return

    logger.info(f"Processing {quarto_doc}")

    base_name = os.path.splitext(os.path.basename(quarto_doc))[0]
    first_part = base_name.split('_')[0]

    for combo in combinations:
        logger.info(f"Processing combination: {combo}")
        
        # Create a dictionary of parameters
        params = dict(zip(parameter_names, combo))
        
        # Apply default assignments
        params = apply_default_assignments(params, default_assignments)
        
        output_label = format_output_label(output_format, params)
        logger.info(f"Formatted output label: {output_label}")
        output_dir = os.path.join("_output", first_part, output_label)
        render_quarto(quarto_doc, output_dir, params)

def main():
    """Main function to orchestrate the Quarto document generation process."""
    parser = argparse.ArgumentParser(description="Generate Quarto documents with parameter combinations.")
    parser.add_argument("quarto_docs", nargs='+', help="Quarto document filenames")
    parser.add_argument("-a", "--all", action="store_true", help="Use render_params_all.yaml instead of render_params.yaml")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    params_file = "render_params_all.yaml" if args.all else "render_params.yaml"
    config = load_yaml(params_file)

    parameter_names = config['parameter_names']
    combinations = expand_combinations(config['parameter_combinations'], parameter_names, config.get('rules', []))
    logger.info(f"Expanded combinations: {combinations}")
    output_format = config['output_format']
    
    # Fetch default assignments from the config
    default_assignments = config.get('default_assignments', [])
    
    logger.info(f"Output format: {output_format}")
    logger.info(f"Parameter names: {parameter_names}")

    for quarto_doc in args.quarto_docs:
        process_quarto_doc(quarto_doc, combinations, parameter_names, output_format, default_assignments)


if __name__ == "__main__":
    main()