#!/bin/bash
cd "$(dirname "$0")"

# Define arrays for parameter names and values
param1_name="train_fb_noise_std"
param2_name="train_motor_noise_std"
param3_name="train_fb_delay_steps"

param1_values=(0.01 0.02 0.04 0.1 0.0 0.0 0.0 0.0)
param2_values=(0.01 0.02 0.04 0.1 0.0 0.0 0.0 0.0)
param3_values=(0 0 0 0 1 2 3 4)

# Make sure all parameter arrays have the same length
if [ ${#param1_values[@]} -ne ${#param2_values[@]} ] || [ ${#param1_values[@]} -ne ${#param3_values[@]} ]; then
    echo "Error: All parameter arrays must have the same length"
    exit 1
fi

# Loop through the parameter values
for i in "${!param1_values[@]}"; do
    param1=${param1_values[i]}
    param2=${param2_values[i]}
    param3=${param3_values[i]}
    
    output_dir="curl_noise-${param1}-${param2}_delay-${param3}"
    
    quarto render 1-2a_analysis-model-perturb.qmd --output-dir "$output_dir" \
        -P "$param1_name:$param1" \
        -P "$param2_name:$param2" \
        -P "$param3_name:$param3"
done
