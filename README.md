# RNNs-learn-robust-policies

## Configuration

Set environment variable `RLRMP_CONFIG_DIR` to the path of your configuration directory.

Then any files in that directory will be loaded instead of the respective files in 
the subpackage `config`, *when loading config files by name*. 
This way you can change the default hyperparameters for an entire part of the experiment.
Default config files, including the ones you modify in your own config directory, need to be complete; i.e. you should not omit any of the keys found in the defaults in the subpackage `config`.

Show how to copy one of the default config files to your own directory, 
e.g. `~/.config/rnns_learn_robust_motor_policies`.

When using the script `train` to train models on tasks, or the script `run_analysis` to run analyses,
the config is passed as a path to a YAML file. 
These config files are allowed to be incomplete, and in fact it may be convenient to use them to specify 
only one or a few parameters that differ from the defaults.
The scripts use the partial configs to update the defaults.