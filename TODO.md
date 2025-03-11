
**See [[results-2|Part 2 results]] for ongoing analysis TODOs.

## Next

- [ ] Aligned vars plotting function for `context_pert` which plots all four conditions (+/- plant pert, +/- context pert) on the same figure (one per training std)

- [ ] [[TODO-analysis#Influence of context input on network dynamics]]
- [ ] [[TODO-analysis#Variation in effective direction, over a reach]]
- [ ] [[TODO-analysis#Individual unit ablation]]

Then: continue with [[TODO-analysis#Network analysis Population level]].

- [ ] [[TODO-analysis#Variation in effective direction, with directional properties of fixed points]]
- [ ] [[TODO-analysis#Stimulation of Jacobian eigenvectors]] 
- [ ] [[TODO-analysis#Steady-state Hessian]]
- [ ] [[TODO-analysis#TMS/tDCS analogues]]
### Convert notebooks

- [ ] 1-2
- [ ] 2-2
- [ ] 2-4
- [ ] 2-5



## Analysis

See [[TODO-analysis]].

## Technical

- [ ] **Convert `model_info` column values to hyperparameters**, so we have access to all the model hps in `hps` without having to fill out the config YAML with whatever we need
	- i.e. properly implement `record_to_namespace`
	- the use of `promote_model_hps` in `flatten_hps` is problematic, since we can’t distinguish column names that refer to model hps, to those that refer to other hps (e.g. `eval_n` or `n_std_exclude`)
	- instead, we should keep the `model__*` prefix for 
- [ ] **Is it really necessary to construct `all_hps` in `run_analysis`?**
	- Consider that we will always have access to the hyperparameter information in the `LDict` levels of `all_models`/`all_tasks`
	- So the question is whether `hps` will ever need to contain information that is specific to a task-model eval pair, aside from the information that is encoded in the structure of the PyTree of pairs
	- However, we would need to modify the code slightly so that `hps` gets updated with the PyTree path info as we map over the leaves
	- Also note that we have `extras` now, which is for more structured info that would not be appropriate to pass 
	- As for the task variant, the differences between the variants is already found in the config file under the YAML key `task`, so we can access it from `hps_common`
- [ ] Combine `Effector_ByEval`, `Effector_SingleEval`, `Effector_ByReplicate`
	- Be careful about axes. For example, in `part1.feedback_perts` we add an impulse amplitude axis; we have to add it at position 2 so as not to interfere with trial/replicate indexing. However, for `Effector_ByReplicate` we will end up lumping the impulse amplitude axis, I think, and coloring by replicate. Is this the intended behaviour?
- [ ] Move the constants out of `constants` and into config files, where possible. Including `REPLICATE_CRITERION`.
- [ ] Add the calculation of `disturbance.amp` when loading hyperparams (e.g. 1-2)

## Figures and formatting 

- [ ] **Optionally add annotations during retrieval** from db
- [ ] **Generate filenames/save to specified directory during retrieval
- [ ] Fix the yaxis range for measure plots in nb 1-2; some of them should display negative values
- [ ] Show trial, replicate, condition info in hoverinfo of individual *aligned* trajectories
- [ ] **Fix `add_endpoint_traces`**. Or was that feature part of the new trajectory plotter? If the latter, update notebooks 1-1 and 1-2 to use the feature

## Training 

**See also [[results-training-methods-part2]].**

- [ ] Debug the equinox vmap warning that keeps showing up at the start of each training run (in 2-1 anyway)
- [ ] Parameter scaleup for part 2
- [ ] **Looks like `train_step` is re-compiling on the first batch, again, since I started passing `batch`. Debug.**
	- Interesting that in the baseline case, the “Training/validation step compiled in…” notices register as ~0 s for the condition (i.e. continuation of baseline) training run, which makes sense given that the identical functions were just compiled for the baseline run; however, there is still a ~5 s delay between when the tqdm bar appears, and when it starts moving!

## Efficiency 


## Database 

- [ ] Switch to a document database like MongoDB (or TinyDB)
	- See [this](https://console.anthropic.com/workbench/279b86f6-39f9-4c42-947f-6f1b02df1224) convo with Claude. 
	- I’m not sure I’ll do this for this project, however it is appealing because it seems to 1) avoid issues with flattening/unflattening hyperparameters, 2) allow more complex queries, 3) doesn’t require an ORM to be defined
- [ ] Function which deletes/archives any .eqx files for which the database record is missing – this will allow us to delete database records to “delete” models, then use this function to clean up afterwards
- [ ] Linking table that maintains foreign key relationships between evaluation records and model records; i.e. currently we store multiple model hashes in a `list[str]` column of the evals table, since an eval can depend on multiple models. But it doesn’t make sense to have an arbitrary number of foreign key columns in this table. Instead, the linking table would have a single entry for each eval-model dependency relation (i.e. a single eval record that refers to the hashes of 5 model records, corresponds to 5 records in the linking table).

###  Converting a record to a `TreeNamespace`

e.g. when loading hps for a model.

Approaches:

1. Split column names to unflatten the record entries into a nested dict, then convert to a namespace. 
2. Use a document database like MongoDB

One approach Claude suggested was to add an extra column that stores the entire hyperparameter JSON, so we can load it all at once without flattening. However, this doesn’t save us anything as we still want to keep the flattened columns for querying purposes, meaning we still need at least `flatten_hps`, and also the database stores all of the hyperparameter values twice. 

### Function to obtain figures based on training + testing conditions

Each figure type also has certain parameters which may vary. We may want a function that tells us which parameters are seen to vary for each figure subtype. This will help with writing more specific queries in the figure table, once we know what train+test conditions we are interested in.

## Publication

- [ ] Make a release of feedbax (dependency reasons)

## Debris 

- [ ] Feedbax: Why is the mean validation loss lower than the mean training loss? Is it because the training task has a different distribution of reach lengths?
- [ ] A `replicate_info` file might sometimes be generated for a non-postprocessed ModelRecord; in that case, its `has_replicate_info` gets set to `1` by `check_model_files` and this results in a “duplicate” error being raised when trying to load the postprocessed model according to its `has_replicate_info` value. It seems this happened when I re-ran `train.py` after it raised an error on `post_training`; i.e. the second time it only ran `post_training` on a single model.
- [ ] Better CLI progress bars
- [ ] Solve: Sometimes `AbstractTask.validation_trials` raises a `jax.errors.UnexpectedTracerError`, which suggests there is a side-effect from a compiled function. Strangely, the backtrace points to the `ticks = jax.vmap(...` line in `feedbax.task`. This might only happen when there is only one validation trial.
- [ ] Separate train and eval seeds in `prng.yml` 
- [ ] Write a `tree_stack` that works with models – can’t just use `apply_to_filtered_leaves` since the shape of the output is changed wrt the input 
- [ ] Try vmapping over `schedule_intervenor`, to get batch dimensions in models and tasks. Batched tasks seems like a missing link to making some of the analyses easier to write.
- [ ] I’m not sure changing the `noise_stds` should be the responsibility of `query_and_load_model`. Otoh, `query_and_load_model` is only used in the scope of this project…
### Equinox warning 

This appears to be related to `train_step` inside of `eqx.filter_value_and_grad` in `feedbax.train`. I tried replacing all the nearby jax.vmap calls with `filter_vmap` but the warning remains

`````
UserWarning: 
Possibly assigning a JAX-transformed callable as an attribute on
equinox._ad._ValueAndGradWrapper. This will not have any of its parameters updated.

For example, the following code is buggy:
```python
class MyModule(eqx.Module):
vmap_linear: Callable

def __init__(self, ...): 
    self.vmap_linear = jax.vmap(eqx.nn.Linear(...))

def __call__(self, ...): 
    ... = self.vmap_linear(...)
```
This is because the callable returned from `jax.vmap` is *not* a PyTree. This means that
the parameters inside the `eqx.nn.Linear` layer will not receive gradient updates.

You can most easily fix this either by applying the wrapper at `__call__` time:
```python
class MyModule(eqx.Module):
linear: Callable

def __init__(self, ...): 
    self.linear = eqx.nn.Linear(...)

def __call__(self, ...): 
    ... = jax.vmap(self.linear)(...)
```
or by using `eqx.filter_vmap` instead (which *does* return a PyTree):
```python
class MyModule(eqx.Module):
vmap_linear: Callable

def __init__(self, ...): 
    self.vmap_linear = eqx.filter_vmap(eqx.nn.Linear(...))

def __call__(self, ...): 
    ... = self.vmap_linear(...)
```
`````

## Archive 

### Best training method  

Out of [[methods#Training methods|these]]. 

> [!warning] 
> This is out of date. I corrected an error in the implementation of PAI. 
> 
> Currently, there are differences between BCS-75 and PAI-ASF, but those differences are not huge.

#### Current state 

- BCS is in some ways the best for constant fields, as behaviour is approximately the same across training stds, for context input 0
- BCS does not even work for curl fields
- DAI works in both cases, but the results show the worst spread of robust behaviour, probably because the network has little uncertainty about how perturbed it will be
- PAI works in both cases, but appears to be [[2024-12-13#PhD|inducing]] a bias/adaptation in the curl field case.

### Motivation for database 

It is kind of annoying to have to store figures in so many subdirectories, and to have to reconfigure the “suffixes” whenever I want to mess with a different hyperparameter. 

Instead, if we stored models and figures using (say) timestamps + hashes, and then saved them as entries in databases (one for models, one for figures?) along with all their hyperparameters, then we could easily filter/query and load/compare them based on those hyperparameters, rather than doing ugly and ungeneralizable parsing of filenames.

It might also make sense to automatically save evaluated states to disk, and to only re-evaluate them if the hash changes for the loaded model (i.e. because we have retrained the same set of hyperparameters, resulting in different trained weigths.) However, this might also be more trouble than it is worth.

### Model-per-file serialisation 

Currently, my model saving and loading process is oriented around entire training runs. 

- All of the models from a training run are saved in the same `.eqx` file, whose name is a hash, which is referenced by a record in the models table of the database. 
- The models (i.e. training run) table has the array-valued column `train__pert__stds`, which is a list of floats. 
- When we load models for analysis, we end up with a `TrainStdDict`, i.e. model records map one-to-one with `TrainStdDict`s.
- It is difficult to load and analyze across the values of hyperparameters other than training disturbance std, e.g. noise level, weight of one of the cost terms, …

*But I have found myself wanting to analyze across noise level, weight of one of the cost terms, …*

#### What would be better? 

- instead of a `TrainStdDict` we will have a `NoiseLevelDict`, `CostWeightDict`, … 
- The model pytree’s type (e.g. `TrainStdDict`) would need to be made explicit in the code for the particular analysis, and we should not explicitly type as `TrainStdDict` in e.g. `query_and_load_models`
- A single model would be serialised per `.eqx` file
- Model records would refer to individual models, and no record of the structure of training runs would be implicitly preserved in the models table.'

##### Advantages 

- Analyses are set up the same way, no matter the hyperparameter being compared across
- The models table is actually the models table
- Can skip re-training models that are already in the db, if we’re doing a new spread that happens to intersect some part of hyperparameter space we’ve already visited

##### Challenges 

Search repo for `TrainStdDict` to find other code that may need to be changed.

- ~~Convert the old “models” table (i.e. load each training run `.eqx` and split it into multiple records with their own `.eqx` files)~~

### Notebooks as scripts 

I want to be able to specify parameters for the entire project, e.g. all the models to train, and which analyses to do on which combinations of models. 

However, while using notebooks (Jupyter, Quarto) it is not so easy to do this. I do not find the Quarto project structure intuitive. I end up running things manually in bits and pieces, which is slow and potentially befuddling.

Instead:

- convert the notebooks to scripts, which can be easily run from other scripts. 
- additional scripts would coordinate the batch runs, loading the project config, etc.

#### Conversion 

Don’t worry about refactoring functions found in the notebooks, for now.  

**Keep the notebooks** for historical/educational purposes.

- [ ] **Convert all notebooks into modules** 
	- Markdown becomes comments as needed, but exclude all lengthy explanations.
	- A bit of refactoring may be helpful. For example, if a notebook contains three major analyses and generates respective sets of figures, then we might have three functions in the converted script that define the individual analyses. The `main` loop calls each of them.

#### Batch scripts 


> [!NOTE]+
> How do we encode the spreads? 
> 
> I want to avoid encoding *all* the parameters, for all the runs. 
> 
> Could stick to the pattern I used for `query_and_load_models`: a single number means the value is constant, whereas a sequence of numbers indicates a spread. All the sequences in a given config, must have the same length (i.e. we are encoding just one spread). 
> 
> Alternatively, we might encode multiple spreads by assuming that any sequence indicates a different dimension of variation. This could make more sense for the configs for the batch scripts, than for the default configs for the individual notebooks. 
> 
> It’s probably fine for this project to stick with single-spread configs. In the batch script configs, we can give a list of multiple single-spread configs (e.g. one single spread over noise, one over delay, one over disturbance std).
> 
> However it would be more expressive to be able to use both types. Is there some way to encode this locally in a YAML (or JSON) file?
#### Challenges 

- it is different to debug scripts, since we have to re-run them each time we run into an error, unless we already have the relevant breakpoint set and are using the debugger 
	- if the code that precedes the error in the script is costly (e.g. slow) then we will suffer that more, maybe.
	- anyway it’s probably fine for analyses which are already (near-)complete

### Hyperparameter data structure 

I am using `SimpleNamespace`, though it might be better to use `namedtuple`.

I can turn `SimpleNamespace` into a PyTree, however this would be automatic with `namedtuple`.

Using `jt.map` is not ideal for replacing null values with defaults, if we allow for partial configs – i.e. if not every config file needs to have all the same keys as all the other config files of the same kind, even if they are assigned `null` and never used – then the pytree structures will not match. 

With `namedtuple`, the user could use partial configs, but needs to specify in the project code the structure of every kind of config file (e.g. training, or analysis). Then the loaded hps would contain unused information. Though, perhaps we could use `namedtuple` and `jt.map` to get the updated hyperparameters, and then just recursively `subdict` out the parameters that appear in the defaults for that kind of config.