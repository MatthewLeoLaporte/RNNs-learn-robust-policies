import os
import re
import yaml
import argparse
from typing import Dict, List, Any, Optional


class FlowList(list):
    pass


def setup_yaml_formatting():
    """Create custom YAML Dumper with desired formatting."""
    class BlockDumper(yaml.Dumper):
        def represent_flowlist(self, data):
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
            
        def represent_list(self, data):
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)
        
        def represent_dict(self, data):
            return self.represent_mapping('tag:yaml.org,2002:map', data, flow_style=False)
            
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)
    
    BlockDumper.add_representer(FlowList, BlockDumper.represent_flowlist)
    BlockDumper.add_representer(list, BlockDumper.represent_list)
    BlockDumper.add_representer(dict, BlockDumper.represent_dict)
    
    return BlockDumper


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


def parse_filename(filename: str, format_string: str) -> Dict[str, str]:
    print(f"Parsing filename: {filename}")

    # Convert format string to regex pattern
    pattern = format_string
    for match in re.finditer(r'\{(\w+)\}', format_string):
        param = match.group(1)
        # Use a capture group to match everything up to underscores and hyphens
        pattern = pattern.replace(match.group(), f'(?P<{param}>[^_]+)')
    
    # Adjust the pattern to allow matching within the filename
    pattern = f".*{pattern}.*"
    
    match = re.search(pattern, filename)
    if not match:
        print("No match found")
        return {}
    
    result = match.groupdict()
    return result


def generate_params_yaml(
    directory: str, 
    file_pattern: str, 
    format_string: str, 
    output_file: str, 
    template_file: Optional[str] = None,
):    
    parsed_param_names = set(re.findall(r'\{(\w+)\}', format_string))

    # Load the template YAML file if provided
    existing_data = {}
    if template_file is not None and os.path.exists(template_file):
        with open(template_file, 'r') as f:
            existing_data = yaml.safe_load(f)

    # Combine parameter names from template and parsed
    template_param_names = set(existing_data.get('parameter_names', []))
    parameter_names = list(template_param_names.union(parsed_param_names))

    all_params = [
        parse_filename(filename, format_string)
        for filename in os.listdir(directory)
        if file_pattern in filename
    ]
    
    parameter_combinations = [
        tuple(params.get(name) for name in parameter_names)
        for params in all_params 
    ]
    
    rules = existing_data.get('rules', [])
    expanded_combinations = []
    for combo in parameter_combinations:
        expanded_combinations.extend(apply_rules(combo, parameter_names, rules))
    parameter_combinations = sorted(
        set(expanded_combinations),
        key=lambda x: [str(v) if v is not None else '' for v in x],
    )

    yaml_data = {
        'parameter_names': parameter_names,
        'parameter_combinations': [FlowList(combo) for combo in parameter_combinations],
        'output_format': existing_data.get('output_format', format_string),
        'rules': existing_data.get('rules', []),
        'default_assignments': existing_data.get('default_assignments', [])
    }

    with open(output_file, 'w') as f:
        yaml.dump(
            yaml_data, 
            f, 
            default_flow_style=None, 
            sort_keys=False,
            indent=2,
            width=float('inf'),
            Dumper=setup_yaml_formatting(),
        )

    print(f"Generated {output_file} with {len(parameter_combinations)} parameter combinations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate params.yaml from filenames.")
    parser.add_argument("format_string", help="Format string to parse filenames")
    parser.add_argument("--directory", default="../models/", help="Directory to search for files")
    parser.add_argument("--file_pattern", default="trained_models.eqx", help="Pattern to match in filenames")
    parser.add_argument("--output", default="render_params_all.yaml", help="Output YAML file (default: render_params_all.yaml)")
    parser.add_argument("--template", default=None, help="Optional existing YAML file to use as template")
    
    args = parser.parse_args()

    generate_params_yaml(args.directory, args.file_pattern, args.format_string, args.output, args.template)