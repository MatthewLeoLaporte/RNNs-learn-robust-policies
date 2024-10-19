"""Generate a YAML file for use with `render.sh`.

Typically, in the `../models` directory, we'll have a number of trained models serialized in 
`.eqx` files. These contain labels in their filenames specifying some important hyperparameters
(e.g. system noise level, disturbance type, and feedback delay). We may want to run some analysis 
notebooks (`.qmd` files) in this directory multiple times, once for each hyperparameter configuration.
For this we use `render.sh`, along with a specification of the sets of parameter values to evaluate 
in `render_params.yaml`. This script generates `render_params_all.yaml` for all the available trained 
models, to make it easy to evaluate one or more notebooks on all available trained models.

Example usage: 

```
python parse_model_filenames.py "_{disturbance_type_load}_noise-{feedback_noise_std_load}-{motor_noise_std_load}_delay-{feedback_delay_steps_load}"
```

Note the explicit underscore at the beginning of the format string. See below for why this is still necessary.
"""

# TODO: Use `logging` rather than print statements

import os
import re
import yaml
import argparse
from typing import Dict, List

def parse_filename(filename: str, format_string: str) -> Dict[str, str]:
    print(f"Parsing filename: {filename}")
    print(f"Using format string: {format_string}")

    # Convert format string to regex pattern
    pattern = format_string
    for match in re.finditer(r'\{(\w+)\}', format_string):
        param = match.group(1)
        # Replace the placeholder with a capture group
        #! This doesn't match the entire string for string-valued parameters; e.g. for "curl" it matches "l",
        #! unless we include an explicit underscore in our format string. 
        #! It should match backward to the first non-alpha character (excluding hyphens; perhaps, the first underscore)
        pattern = pattern.replace(match.group(), f'(?P<{param}>[^_-]+)')
    
    # Add .* at the start and end to allow matching anywhere in the filename
    pattern = f".*{pattern}.*"
    print(f"Final regex pattern: {pattern}")
    
    match = re.search(pattern, filename)
    if not match:
        print("No match found")
        return {}
    
    result = match.groupdict()
    print(f"Parsed parameters: {result}")
    return result

def generate_params_yaml(directory: str, file_pattern: str, format_string: str, output_file: str):
    parameter_names = re.findall(r'\{(\w+)\}', format_string)
    parameter_combinations = []

    print(f"Searching for files matching '{file_pattern}' in '{directory}'")
    print(f"Using format string: {format_string}")
    print(f"Extracted parameter names: {parameter_names}")

    for filename in os.listdir(directory):
        if file_pattern in filename:
            print(f"\nProcessing file: {filename}")
            params = parse_filename(filename, format_string)
            if params:
                combination = [params[name] for name in parameter_names]
                print(f"Extracted combination: {combination}")
                if combination not in parameter_combinations:
                    parameter_combinations.append(combination)

    # Sort parameter combinations for consistency
    parameter_combinations.sort()

    yaml_data = {
        'parameter_names': parameter_names,
        'parameter_combinations': parameter_combinations,
        'output_format': format_string
    }

    with open(output_file, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=None, sort_keys=False)

    print(f"\nGenerated {output_file} with {len(parameter_combinations)} parameter combinations.")
    print(f"Output format: {format_string}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate params.yaml from filenames.")
    parser.add_argument("format_string", help="Format string to parse filenames")
    parser.add_argument("--directory", default="../models/", help="Directory to search for files")
    parser.add_argument("--file_pattern", default="trained_models.eqx", help="Pattern to match in filenames")
    parser.add_argument("--output", default="render_params_all.yaml", help="Output YAML file (default: render_params_all.yaml)")
    
    args = parser.parse_args()

    generate_params_yaml(args.directory, args.file_pattern, args.format_string, args.output)