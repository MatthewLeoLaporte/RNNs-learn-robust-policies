
**See [[results-2|Part 2 results]] for ongoing analysis TODOs.

- [ ] Feedbax: Why is the mean validation loss lower than the mean training loss? Is it because the training task has a different distribution of reach lengths?
- [ ] A `replicate_info` file might sometimes be generated for a non-postprocessed ModelRecord; in that case, its `has_replicate_info` gets set to `1` by `check_model_files` and this results in a “duplicate” error being raised when trying to load the postprocessed model according to its `has_replicate_info` value. It seems this happened when I re-ran `train.py` after it raised an error on `post_training`; i.e. the second time it only ran `post_training` on a single model.
- [ ] Use dunder for `join_with` in `flatten_hps`, uncomment the lines at the start of `run_analysis.setup_tasks_and_models`, and restart the DB

## Next

- [ ] [[#Influence of context input on network dynamics]]
- [ ] [[#Variation in effective direction, over a reach]]
- [ ] [[#Individual unit ablation]]

Then: continue with [[#Network analysis Population level]].

- [ ] [[#Variation in effective direction, with directional properties of fixed points]]
- [ ] [[#Stimulation of Jacobian eigenvectors]] 
- [ ] [[#Steady-state Hessian]]
- [ ] [[#TMS/tDCS analogues]]

### Technical

- [ ] `seed`/base `key` column in each of the db tables
- [ ] Move the constants out of `constants` and into config files, where possible. Including `REPLICATE_CRITERION`.
- [ ] **In `types`, make the mapping algorithmic between custom dict types and the column names they map to. Thus `PertVarDict` keys correspond to `pert_var` column values.**
	- **Use `LevelDict`**
	- Similarly, we can use the same system to automatically determine the axis labels for `get_violins`
- [ ] **In `AbstractAnalysis.save_figs`, format the dump filename based on `path_params` and not just an integer counter**
- [x] Make sure we save *all* analysis figures in `AbstractAnalysis.save_figs`; currently we’re only saving a subset for `Measures_CompareTrainStdAndContext`, it seems
	- This was because of an incorrect change I had just made to `save_figs`, which is now reverted
- [ ] Add the calculation of `disturbance.amplitude` when loading hyperparams (e.g. 1-2)

### Convert notebooks for part 1

- [ ] 1-2
- [x] 1-3
- [x] 2-1
- [ ] 2-2
- [ ] 2-4
- [ ] 2-5

## Analysis

- [ ] Try training a significantly larger network in part 2 and see if the context 0 curves overlap more

### Behaviour 

- [ ] See how much more robust the baseline model is in part 1, if we **decrease the weight on the control cost**. (Will also affect the time urgency.)
	- My expectation is that it will of course increase the forces and decrease the movement duration, but that depending on the perturbation this will actually make it *less* robust (e.g. curl fields)
- [ ] In part 2, what is the relationship between train std. and robustness, at a given context input? 

#### Measures

- [ ] **Direction of max net force**
	- at start of trial?

### Network analysis: Unit level

> [!NOTE]
> Using part 2 hybrid models, since otherwise there is no basis from comparison between more- and less-robust networks.

#### Influence of context input on network dynamics 

##### Steady-state

- [ ] **Run a steady-state trial and change the context input (but nothing else) midway through**
	- or, ramp the context input across the trial
	- how does the activity change?
	- does the point mass move? hopefully not, but the network was never trained with within-trial variation of the context

##### Reaching

Repeat but change the context input during a reach, either step or ramp

#### Preferred versus effective directions of units at steady state 

> [!warning] 
> Make sure where the intervention is taking place in Feedbax! Do we perturb before or after the recurrent update? If after, then the immediate consequence of stimulation is entirely determined by the readout, which isn’t very interesting. But we’ll still get the recurrent effects on the next step, of course.


> [!NOTE] 
> Also we can analyze these two individually at steady state, but the preferred directions on their own don’t tell us much.)
> 
> - [ ] **Do tuning curves get narrower** (i.e. do units become less active more quickly as they move away from their preferred direction?) with context input?

Here, our analysis is based on evaluation on two tasks.

1. At the origin steady state, perturb the point mass with a constant force for several time steps. Infer unit preferred directions from their activities at peak force output for a trial.
2. Also at the origin steady state, perturb a network unit with a constant input for several time steps. Repeat for each unit in the network. Infer effective directions as 

##### Preferred directions analysis

- At steady state, perturb the point mass in center-out directions
- At the time step of max forward force (max accel.), find the activities of all units
- Find instantaneous preferences ~~(direction of max activity for each unit; circular distributions of activity for each unit)~~

> [!NOTE] 
> Preferences should not be just argmax, especially since this limits our resolution by the number of directions we perturb in. 
> 
> Instead, use regression to infer a vector in the direction of max activity

##### Effective directions single-unit stimulation analysis

- At steady state, perturb each unit in the network
- Compare the direction of max acceleration, to the preferred direction of the same unit from 1. Note that the instantaneous preferences may not capture the unit-specific effects on recurrent processing, whereas unit perturbations should induce them.
- Do the preferred & perturbed directions align more, for low vs. high context inputs? 
- Also can do other analyses on the perturbed responses...

#### Variation in effective direction, over a reach 

Repeat the effective direction single-unit stimulation analysis, separately at several time points along a reach. 
How do the (distributions of) effective directions of individual units shift over the reach? 
Does this change with context?

- [ ] Also: Variation in preferred versus effective direction, over a reach
- [ ] 
#### Individual unit ablation 

Fix the activity of each unit to zero, in turn.

- [ ] Is performance more sensitive to the ablation of some units, than others? 
- [ ] How does this depend on reach direction? etc?

### Network analysis: Population level

- [ ] Also: repeat all of this for a leaky vanilla RNN

#### Troubleshooting FP finding 

- [ ] **Examine the loss curve over iterations of the fixed point optimization**
	- Get an idea how quickly it converges, if this varies much between conditions, and if there is something to be understood about the FPs that (appear to) fail to converge
	- possibly, plot the hidden state over the optimization, like I did once before (when I showed that the multiple FPs found by the algorithm were actually corresponding to a single FP, by allowing them to converge more)
- [ ] Try to get rid of dislocations in fixed point trajectories (though they aren’t very bad except at context -2)

#### Translation (in)variance 

- [x] Do fixed points vary with goal position (i.e. target input), or just with the difference between target and feedback inputs?
	- for example, do the fixed points change if we translate the target and feedback position inputs in the same way?
	- The “ring” of steady-state (“goal-goal”) FPs for simple reaching suggests this might be the case. 
	- **They do vary.** The grid of steady-state FPS form a planar grid in the space of the top three PCs. As context input changes, this grid translates along an ~orthogonal direction. 
	- **Should plot the readout vector** and see if the direction of translation is roughly aligned with it, which would make sense. 

#### Steady-state FPs 

##### Variation with feedback inputs

i.e. find steady-state FPs, then change the feedback input and see if the location/properties of the FP changes

- [ ] As we increase the perturbation size, do the eigenvectors become aligned? Do they point in the direction in state space that would increase the readout in the corrective direction?
- [ ] **Identify the most unstable eigenvectors. Which units have the strongest contribution?** If we perturb these units at steady state, versus some other randomly selected set of units, do we get a large “corrective” change in the network output?

##### Variation with context input

- [ ] **How do the init-goal and goal-goal fixed point structures vary with context input?**

#### Steady-state Jacobian eigenspectra 

- [ ] What are the “wing” eigenvalues, e.g. seen for the origin steady-state FP, in DAI+curl?
	- Note that they seem to be the strongest oscillatory modes
	- They become larger (i.e. decay more slowly) and higher-frequency with increasing context input 
	- They become relatively larger and higher-frequency when training on stronger field std.

##### Stimulation of Jacobian eigenvectors

- [ ] At the origin steady state FP, perturb the eigenvectors of the Jacobian

When we stimulate them what is different from when we stimulate one of the eigenvectors whose eigenvalue is in the circle around 0? 
How does this change based on context input? e.g. the “circle around origin” doesn’t really exist for the less-robust networks.

#### Steady-state Hessian 

- Are the fixed points minima of the dynamics? 
- Look at the Hessian eigenspectra
- Is there a difference between steady and unsteady FPs? e.g. maybe steady FPs appear as minima, but unsteady ones less so?

#### Variation in effective direction, with directional properties of fixed points 

Repeat the [[#Effective directions single-unit stimulation analysis]] at FPs which have a direction associated with them.
Now, each of these FPs is associated with a non-zero force in some direction. 
How does that direction relate the to the effective direction? 
Do effective directions bend towards the FP force direction? 
Does this happen more or less, in more robust networks?


> [!NOTE]
> This is in this section on population (versus unit) analysis because while it is a single-unit analysis, it is investigating population properties (i.e. of the FP) as well. 
> 
> Thus we can present the single-unit results, then the FPs, then the results that depend on both.

##### Reaches

In the middle of reach trajectories are “unsteady” fixed points which correspond to non-zero force outputs, and which the network usually only approaches and does not pass through.

Here, it might only make sense to do the unit perturbation analysis for a single time step (assuming we perturb before the recurrent update).

##### Static loads

When the point mass has to remain stationary under a static force, the network will be at steady state at a fixed point corresponding to an equal and opposite force. 

Thus, place the point mass at the origin, make its goal the origin, and apply a load it has to statically counter. When it reaches steady state again, perform the effective direction.

#### TMS/tDCS analogues 

Stimulate all the units in the network simultaneously, to mimic TMS/tDCS. Specifically:

- TMS is analogous to adding to the hidden vector (i.e. directly increasing the firing rate)
- tDCS is analogous to adding a small bias term (i.e. in Feedbax, add to the hidden state *after* doing the recurrent update, but before the nonlinearity)

How does this vary with context input? If there are clear differences, then we may be able to predict what will happen when we apply TMS/tDCS.

#### Other 
##### Poincare graph - phase portraits

![[file-20250116174358466.png]]

- [x] see work scratch
	- we graph $\det{A}$ versus $\mathrm{Tr}~A$
	- (but this is for continuous systems I think)
- http://phaseportrait.blogspot.com/2013/01/discrete-time-phase-portraits.html
- for LTI systems, it may make sense to do the discrete → continuous conversion
- https://en.wikipedia.org/wiki/Logarithm_of_a_matrix#Calculating_the_logarithm_of_a_diagonalizable_matrix

If this does work, then:

- [ ] See how context input (and train std? reach directions?) shifts the plot

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

- [x] Maybe save a separate model table record in the database, for each *separate* disturbance std during training. This will:
	- Prevent us from saving the same trained model multiple times because e.g. once we trained three runs `[0, 0.1, 1]` and another time two runs `[0, 0.1]`.
	- Force us to use the multi-loading logic in the analyses notebooks (i.e. load db entry with std 0, and db entry with std 0.1, and compare); note that this kind of logic will be necessary anyway for some of the supplementary analyses (e.g. noise comparisons).
	- Note that the foreign refs in the eval and fig tables would change into sets/lists, which is probably as it should be.
- [x] Make wrapper for `go.Figure` that adds a “copy as png” and “copy as svg” buttons; this will save a lot of time
	- See https://plotly.com/python/figurewidget-app/

## Database 

- [ ] Function which deletes/archives any .eqx files for which the database record is missing – this will allow us to delete database records to “delete” models, then use this function to clean up afterwards
- [ ] Linking table that maintains foreign key relationships between evaluation records and model records; i.e. currently we store multiple model hashes in a `list[str]` column of the evals table, since an eval can depend on multiple models. But it doesn’t make sense to have an arbitrary number of foreign key columns in this table. Instead, the linking table would have a single entry for each eval-model dependency relation (i.e. a single eval record that refers to the hashes of 5 model records, corresponds to 5 records in the linking table).

#### Function to obtain figures based on training + testing conditions

Each figure type also has certain parameters which may vary. We may want a function that tells us which parameters are seen to vary for each figure subtype. This will help with writing more specific queries in the figure table, once we know what train+test conditions we are interested in.

## Debris 

- [ ] Construction of the analysis graph might be too complicated; is there a way to make analysis classes cache their results for a particular input, without causing problems with JAX?
- [ ] Better CLI progress bars
- [ ] Solve: Sometimes `AbstractTask.validation_trials` raises a `jax.errors.UnexpectedTracerError`, which suggests there is a side-effect from a compiled function. Strangely, the backtrace points to the `ticks = jax.vmap(...` line in `feedbax.task`. This might only happen when there is only one validation trial.
- [x] Double check that `save_model_and_add_record` is detecting existing records – not just generating a new hash and save a new model even if all the hps match
- [ ] Separate train and eval seeds in `prng.yml` 
- [ ] Write a `tree_stack` that works with models – can’t just use `apply_to_filtered_leaves` since the shape of the output is changed wrt the input 
- [ ] Try vmapping over `schedule_intervenor`, to get batch dimensions in models and tasks. Batched tasks seems like a missing link to making some of the analyses easier to write.
- [ ] Stop using `tmp` in figures dir
- [ ] I’m not sure changing the `noise_stds` should be the responsibility of `query_and_load_model`. Otoh, `query_and_load_model` is only used in the scope of this project afaik…
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
- The models (i.e. training run) table has the array-valued column `disturbance_stds`, which is a list of floats. 
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

- [x] ~~Update `query_and_load_models`~~: currently we pass the values of some model record fields, and check them one-to-one to find a match. Thus `disturbance_std_load` is a list of float, which is matched exactly with a list of float in the “models” table. **Instead, any sequences found in `query_and_load_models` should be interpreted as “map over the models table, loading multiple records, over this spread of values”.** For now, I think it is sufficient that we assume that if any lists are present, they are all the same length, and that we’re only loading a single spread (though multiple hyperparameter values may vary across individual models). 
	- **No, this shouldn’t be in `query_and_load_models`**. Instead, we should query and return a single model, and then map over this function in the notebook itself. Thus any logic about loading/handling multiple models for the purpose of an analysis, is done in the notebook.
- [x] `setup_task_model_pairs` should only take a single `disturbance_std`, and should not return a `TrainStdDict`
- [x] Records in the eval table correspond to runs of an analysis notebook. The analysis may be across several single models. Currently, the eval and figure records each have a single foreign reference to the hash of the training run (i.e. model record). Instead, we would need several such references. I don’t think these can be kept as multiple foreign SQL keys. ~~**Instead, we may need some kind of [linking table](https://stackoverflow.com/a/20572207).**~~ (I’m not doing this, for now.) A simpler solution is just store the model hashes in a list-of-string column, and forego the foreign key features, which are mostly a convenience at this point.
- ~~Convert the old “models” table (i.e. load each training run `.eqx` and split it into multiple records with their own `.eqx` files)~~
- [x] Update the main loop in `post_training`

### Notebooks as scripts 

I want to be able to specify parameters for the entire project, e.g. all the models to train, and which analyses to do on which combinations of models. 

However, while using notebooks (Jupyter, Quarto) it is not so easy to do this. I do not find the Quarto project structure intuitive. I end up running things manually in bits and pieces, which is slow and potentially befuddling.

Instead:

- convert the notebooks to scripts, which can be easily run from other scripts. 
- additional scripts would coordinate the batch runs, loading the project config, etc.

#### Conversion 

Don’t worry about refactoring functions found in the notebooks, for now.  

**Keep the notebooks** for historical/educational purposes.

- [x] Decide between argparse, and `main` functions that take a bunch of parameters (probably the latter.)
- [ ] **Convert all notebooks into modules** 
	- Markdown becomes comments as needed, but exclude all lengthy explanations.
	- A bit of refactoring may be helpful. For example, if a notebook contains three major analyses and generates respective sets of figures, then we might have three functions in the converted script that define the individual analyses. The `main` loop calls each of them.

#### Batch scripts 

- [x] Add YAML files indicating the spreads of models to train for part 1 and 2
- [x] Add batch script which runs a given script based on a given YAML file 
	- For that, we need a common interface, i.e. we won’t need a different batch script for part 1 versus part 2.

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