#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.
cd "$(dirname "$0")"  # Change to the directory of the script

# Default values
all_flag=false
params_file="render_params.yaml"

# Parse command line options
while getopts ":a" opt; do
  case ${opt} in
    a )
      all_flag=true
      params_file="render_params_all.yaml"
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))  # Remove the options from the positional parameters

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide at least one Quarto document filename as an argument."
    echo "Usage: $0 [-a] <quarto_document1.qmd> [quarto_document2.qmd ...]"
    echo "  -a    Use render_params_all.yaml instead of render_params.yaml"
    exit 1
fi

# Check if yq is installed
if ! command -v yq &> /dev/null; then
    echo "Error: yq is not installed. Please install it to parse YAML files."
    exit 1
fi

# Check if the parameters file exists
if [ ! -f "$params_file" ]; then
    echo "Error: Parameters file '$params_file' not found."
    exit 1
fi

# Load parameter names and output format
mapfile -t param_names < <(yq e '.parameter_names[]' "$params_file")
output_format=$(yq e '.output_format' "$params_file")

# Get the number of parameter combinations
num_combinations=$(yq e '.parameter_combinations | length' "$params_file")

# Function to format output directory name
format_output_dir() {
    local format="$1"
    shift
    local params=("$@")
    
    for ((i=0; i<${#param_names[@]}; i++)); do
        local name="${param_names[i]}"
        local value="${params[i]}"
        format="${format//\{$name\}/$value}"
    done
    
    echo "$format"
}

# Loop through each provided filename
for quarto_doc in "$@"; do
    # Check if the file exists
    if [ ! -f "$quarto_doc" ]; then
        echo "Warning: The file '$quarto_doc' does not exist. Skipping."
        continue
    fi

    echo "Processing $quarto_doc"

    # Extract the first part of the filename (before the first underscore)
    base_name=$(basename "$quarto_doc" .qmd)
    first_part=${base_name%%_*}

    # Loop through the parameter combinations
    for ((i=0; i<num_combinations; i++)); do
        params=""
        param_values=()
        
        # Build the parameter string and collect parameter values
        for ((j=0; j<${#param_names[@]}; j++)); do
            param_name=${param_names[j]}
            param_value=$(yq e ".parameter_combinations[$i][$j]" "$params_file")
            params+=" -P \"$param_name:$param_value\""
            param_values+=("$param_value")
        done
        
        # Format the output directory name
        formatted_output=$(format_output_dir "$output_format" "${param_values[@]}")
        output_dir="_output/$first_part/$formatted_output"
        
        # Run quarto render with the built parameter string
        eval quarto render "$quarto_doc" --output-dir "$output_dir" $params
    done
done