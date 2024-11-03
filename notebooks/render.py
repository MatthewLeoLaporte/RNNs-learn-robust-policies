"""Run batches of `quarto render` commands.

Written with the help of Claude 3.5 Sonnet.
"""

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
        actions = rule['action']
        
        conditions_met = (
            combination[parameter_names.index(param_name)] == str(required_value)
            for param_name, required_value in rule['condition'].items()
            # if param_name in parameter_names
        )
        
        if all(conditions_met):
            new_combo = tuple(
                str(rule['action'][param]) if param in rule['action'] else value
                for param, value in zip(parameter_names, combination)
            )
            new_combinations.append(new_combo)
        
    return new_combinations


def apply_default_assignments(params: Dict[str, Any], default_assignments: List[Dict[str, str]]) -> Dict[str, Any]:
    """Apply default assignments to the parameters when they are null."""
    for assignment in default_assignments:
        for target_key, source_key in assignment.items():
            if params.get(target_key) is None:
                params[target_key] = params.get(source_key)
    return params


def expand_combinations(
    combinations: list[list[Any]], parameter_names: list[str], rules: list[dict[str, Any]]
) -> list[tuple[Any]]:
    """Expand all combinations by applying rules."""
    expanded = []
    for combo in combinations:
        expanded.extend(apply_rules(combo, parameter_names, rules))
    return list(set(tuple(x) for x in expanded))  # Remove duplicates


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
    

def process_combination(quarto_docs: List[str], combo: List[Any], parameter_names: List[str], 
                       output_format: List[List[str]], default_assignments: List[Dict[str, str]], 
                       output_base_dir: str = "_output"):  # Add output_base_dir parameter
    """Process all Quarto documents with a single parameter combination."""
    logger.info(f"Processing combination: {combo}")
    
    # Create a dictionary of parameters
    params = dict(zip(parameter_names, combo))
    
    # Apply default assignments
    params = apply_default_assignments(params, default_assignments)
    
    for quarto_doc in quarto_docs:
        if not os.path.exists(quarto_doc):
            logger.warning(f"The file '{quarto_doc}' does not exist. Skipping.")
            continue
            
        logger.info(f"Processing {quarto_doc}")
        base_name = os.path.splitext(os.path.basename(quarto_doc))[0]
        first_part = base_name.split('_')[0]
        
        output_label = format_output_label(output_format, params)
        logger.info(f"Formatted output label: {output_label}")
        output_dir = os.path.join(output_base_dir, first_part, output_label)  # Use output_base_dir instead of hardcoded "_output"
        render_quarto(quarto_doc, output_dir, params)
        
def main():
    """Main function to orchestrate the Quarto document generation process."""
    parser = argparse.ArgumentParser(description="Generate Quarto documents with parameter combinations.")
    parser.add_argument("quarto_docs", nargs='+', help="Quarto document filenames")
    parser.add_argument("-a", "--all", action="store_true", help="Use render_params_all.yaml instead of render_params.yaml")
    parser.add_argument("-o", "--output-dir", default="_output", help="Base output directory (default: _output)")  # Add argument
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    params_file = "render_params_all.yaml" if args.all else "render_params.yaml"
    config = load_yaml(params_file)

    parameter_names = config['parameter_names']
    combinations = expand_combinations(config['parameter_combinations'], parameter_names, config.get('rules', []))
    logger.info(f"Expanded combinations: {combinations}")
    output_format = config['output_format']
    default_assignments = config.get('default_assignments', [])
    
    logger.info(f"Output format: {output_format}")
    logger.info(f"Parameter names: {parameter_names}")

    # Process each combination for all documents
    for combo in combinations:
        process_combination(args.quarto_docs, combo, parameter_names, output_format, 
                          default_assignments, args.output_dir)  # Pass output_dir argument
        
if __name__ == "__main__":
    main()