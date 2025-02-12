---
created: 2024-10-04T11:27
updated: 2024-11-13T10:32
---
**See [[results-2|Part 2 results]] for ongoing analysis TODOs.

- [ ] Try training a significantly larger network in part 2 and see if the context 0 curves overlap
- [ ] Use rich progress bar for CLI?
- [x] **Point mass damping**

### Poster abstract 

> - Ensure your NCM membership is in good standing
> - Select a Presentation Theme that most closely describes your presentation (see list below)
> - Prepare to enter your contact name, affiliation/institution name and email address
> - Prepare an **abstract title** (max 225, incl. spaces, character limit)
> - Enter **three (3) highlights** of the submission (max 100 characters per highlight)
> - Include a **justification statement** on why the presentation is relevant to NCM or would be interesting to have as part of the program (max 1300 characters)
> - Prepare the **abstract** (max 3000 character limit, incl. spaces – approximately **500 words**)
> - Submit the abstract via NCM’s online abstract management system. Please note, when submitting the abstract cut and paste it from a text editor (ie. notepad, wordpad) to ensure it does not include underlying formatting that may cause errors
>   
>   Themes:
> *Control of Eye and Head Movement  
> Posture and Gait  
> Fundamentals of Motor Control  
> Integrative Control of Movement  
> Disorders of Motor Control  
> Adaption and Plasticity in Motor Control  
> **Theoretical and Computational Motor Control**

### Unit perturbations

See: [[#Individual unit stimulation]].

#### Steady-state

Start with this. 

At steady state, what is the distribution of unit preferences?

How does this depend on the steady-state position? 

#### Unsteady state

How does the distribution of unit preferences at a fixed point, depend on the direction of the position/velocity error?

i.e. “at this FP we are not at steady state, we are outputting a force in some direction to try to reach steady state; in this state, what happens if I perturb a unit?”

This might not make sense, depending on whether we can simply look at instantaneous tuning or not, since we will quickly move away from the FP as multiple time steps pass. 
#### Reaching

Here, it would be interesting to see how the distribution of preferred directions changes over the trial. 

Presumably it won’t change much if we only look at the instantaneous tuning, since it will mostly depend on the readout.

However, more generally we might expect that more units are tuned in the direction of the current reach near the beginning of the trial, and then in the opposite direction toward the end. 

### Convert notebooks for part 1

- [ ] **Convert `COLORSCALES` to a `TreeNamespace`, so its structure reflects that of `hps`.**
- [ ] **Move `analysis.part1` and `analysis.part2` and the part1 and part 2 files into `config` subpackage**
- [ ] Move the constants out of `constants` and into config files, where possible. Including `REPLICATE_CRITERION`.
- [ ] **Convert 1-2, and move any shared functions out of 1-1 and into `analysis` or something**
- [ ] In `types`, make the mapping algorithmic between custom dict types and the column names they map to. Thus `PertVarDict` keys correspond to `pert_var` column values.
- [ ] Add the calculation of `disturbance.amplitude` when loading hyperparams (e.g. 1-2)

#### Hyperparameter data structure

I am using `SimpleNamespace`, though it might be better to use `namedtuple`.

I can turn `SimpleNamespace` into a PyTree, however this would be automatic with `namedtuple`.

Also, using `jt.map` is not ideal for replacing null values with defaults, if we allow for partial configs (i.e. if not every config file needs to have all the same keys as all the other config files of the same kind, even if they are assigned `null` and never used) then the pytree structures will not match. 

With `namedtuple`, the user could use partial configs, but needs to specify in the project code the structure of every kind of config file (e.g. training, or analysis). Then the loaded hps would contain unused information. Though, perhaps we could use `namedtuple` and `jt.map` to get the updated hyperparameters, and then just recursively `subdict` out the parameters that appear in the defaults for that kind of config.

#### Task and model setup

Done, plus or minus any minor bugs I haven’t noticed yet.

#### Individual analyses

Some of these may depend on certain things we’d like to only calculate once.
For example, aligned responses. 
Either we add in one or more extra calculation phases where we can construct an “extra data” namespace which is passed around to the individual analyses. 

## Efficiency

- [x] Maybe save a separate model table record in the database, for each *separate* disturbance std during training. This will:
	- Prevent us from saving the same trained model multiple times because e.g. once we trained three runs `[0, 0.1, 1]` and another time two runs `[0, 0.1]`.
	- Force us to use the multi-loading logic in the analyses notebooks (i.e. load db entry with std 0, and db entry with std 0.1, and compare); note that this kind of logic will be necessary anyway for some of the supplementary analyses (e.g. noise comparisons).
	- Note that the foreign refs in the eval and fig tables would change into sets/lists, which is probably as it should be.
- [x] Make wrapper for `go.Figure` that adds a “copy as png” and “copy as svg” buttons; this will save a lot of time
	- See https://plotly.com/python/figurewidget-app/

### Notebooks as scripts

I want to be able to specify parameters for the entire project, e.g. all the models to train, and which analyses to do on which combinations of models. 

However, while using notebooks (Jupyter, Quarto) it is not so easy to do this. I do not find the Quarto project structure intuitive. I end up running things manually in bits and pieces, which is slow and potentially befuddling.

Instead:

- convert the notebooks to scripts, which can be easily run from other scripts. 
- additional scripts would coordinate the batch runs, loading the project config, etc.

#### Conversion

Don’t worry about refactoring functions found in the notebooks, for now.  

**Keep the notebooks** for historical/educational purposes; however they **may go out of sync** with the scripts.

- [x] Decide between argparse, and `main` functions that take a bunch of parameters (probably the latter.)
- [ ] Convert notebooks into modules
	- Markdown becomes comments as needed, but exclude all lengthy explanations.
	- A bit of refactoring may be helpful. For example, if a notebook contains three major analyses and generates respective sets of figures, then we might have three functions in the converted script that define the individual analyses. The `main` loop calls each of them.

#### Batch scripts

- [ ] Add YAML files indicating the spreads of models to train for part 1 and 2
- [ ] Add batch script which runs a given script based on a given YAML file
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

## Analysis

### Behaviour

- [ ] See how much more robust the baseline model is in part 1, if we **decrease the weight on the control cost**. (Will also affect the time urgency.)
	- My expectation is that it will of course increase the forces and decrease the movement duration, but that depending on the perturbation this will actually make it *less* robust (e.g. curl fields)
- [ ] In part 2, what is the relationship between train std. and robustness, at a given context input? 
	- [ ] Across context inputs?
	- [ ] …
- [ ] ~~For each train std. in the “std” method, there is a certain **context input at which the position profile is closest to a straight reach**. Determine what this context input is.~~ (I’m not sure about error bounds; I guess we would do the optimization for *all* validation trials individually.)
#### Measures

- [ ] **Direction of max net force**
	- at start of trial?

### Network perturbations

#### Individual unit stimulation

Perturb the activity of units in the network, one at a time:

- [ ] at steady state
- [ ] during reaches

If during reaches, then we could stimulate at multiple times during the reach, to see how tuning changes.

Example methods:

- Perturb a single unit in all the different contexts (e.g. force field strength), resulting in a bunch of different responses for different contexts/context combinations for a single unit
- Observe qualitatively what changes between context. For example, if context only changes the amplitude of the stimulation response but not the direction, then we will boil each response (set of states) to a single number. So we will have one number (e.g. relative amplitude) for each context/context combination; i.e. N numbers for N contexts.
- Do e.g. linear regression to turn N numbers to M numbers, where M is the number of context variables (i.e. get trends for each context variable)
- Repeat this for all the other units
- Now we can e.g. do a scatter plot of the regression parameters across all the units
- **Are we sticking with a GRU? I assume we are stimulating the candidate vector but the others could be interesting as well.**

> [!NOTE]
> One thing I'm unsure about: the very first step of stimulation, will that only reveal the readout? 
> i.e. because there’s no time for recurrent activity so we’ll basically just project out the extra activity in the stimulated unit
> 
> Double check this is the case: what is the comp graph between a unit’s activity, and 


##### Non-zero-force steady states

**There are also non-zero-force steady states (e.g. the steady states in constant-field trials).**

- If you apply a static force to the limb, and the network has to output a constant force to maintain steady state
- The perturbation responses are probably state-dependent
- Steve has done this sort of thing; Gunnar thinks this may reveal something about… gain modulation? But he seemed unsure

#### Individual unit ablation

Fix the activity of each unit to zero, in turn.

- [ ] Is performance more sensitive to the ablation of some units, than others? 
- [ ] How does this depend on reach direction? etc?

#### Eigenvectors

- [ ] At steady state, perturb the network by the eigenvectors of its Jacobian 
	- [ ] Hessian?

Run this in different positions in the workspace to see if it is state dependent

### Fixed points

- [ ] **Hessians** - describe the curvature
- [ ] Examine reaching FP trajectories for both baseline (no disturbance) and disturbance conditions
- [ ] Try a leaky vanilla RNN

#### Troubleshooting

- [ ] **Examine the loss curve over iterations of the fixed point optimization**
	- Get an idea how quickly it converges, if this varies much between conditions, and if there is something to be understood about the FPs that (appear to) fail to converge
	- possibly, plot the hidden state over the optimization, like I did once before (when I showed that the multiple FPs found by the algorithm were actually corresponding to a single FP, by allowing them to converge more)
- [ ] Try to get rid of dislocations in fixed point trajectories (though they aren’t very bad except at context -2)

#### Translation invariance

- [x] Do fixed points vary with goal position (i.e. target input), or just with the difference between target and feedback inputs?
	- for example, do the fixed points change if we translate the target and feedback position inputs in the same way?
	- The “ring” of steady-state (“goal-goal”) FPs for simple reaching suggests this might be the case. 
	- **They do vary.** The grid of steady-state FPS form a planar grid in the space of the top three PCs. As context input changes, this grid translates along an ~orthogonal direction. 
	- **Should plot the readout vector** and see if the direction of translation is roughly aligned with it, which would make sense. 
- [ ] ~~(In other words?:) Is there only a single steady-state fixed point for each trained network (+context input)?~~
	- If not, do all steady-state FPs show the same kinds of behaviour (e.g. increasing context input → more contracting?)
	- **Based on the eigenspectra**, the steady-state grid FPs are approximately similar in their dynamical properties. There appears to be significantly more variation between context inputs, than between positions. If we zoom in on a particular eigenvalue, it appears to shift its position in tiny ~grids reflecting the change in position.

#### Network input perturbations

i.e. find steady-state FPs, then change the feedback input and see if the location/properties of the FP changes

- [ ] As we increase the perturbation size, do the eigenvectors become aligned? Do they point in the direction in state space that would increase the readout in the corrective direction?
- [ ] **Identify the most unstable eigenvectors. Which units have the strongest contribution?** If we perturb these units at steady state, versus some other randomly selected set of units, do we get a large “corrective” change in the network output?

#### FP trajectories

How do the fixed points change over reach/stabilization trajectories? 

- [ ] **How do the init-goal and goal-goal fixed point structures vary with context input?**
- [ ] **What about the init-goal fixed points for a feedback perturbation (i.e. if we suddenly change the network input)?**

#### Hessian analysis

- [ ] Demonstrate that the fixed points are dynamical minima (vs. say saddle-points)? (Positive definite?)
- [ ] Does the curvature change with training std? Context input?
- [ ] Think about what the eigenspectra might show, here. Not the directions of fastest movement, but of fastest acceleration.

#### Jacobian eigenspectra

- [ ] What are the “wing” eigenvalues, e.g. seen for the origin steady-state FP, in DAI+curl?
	- Note that they seem to be the strongest oscillatory modes
	- They become larger (i.e. decay more slowly) and higher-frequency with increasing context input 
	- They become relatively larger and higher-frequency when training on stronger field std.
#### Poincare graph - phase portraits

![[file-20250116174358466.png]]

- [x] see work scratch
	- we graph $\det{A}$ versus $\mathrm{Tr}~A$
	- (but this is for continuous systems I think)
- http://phaseportrait.blogspot.com/2013/01/discrete-time-phase-portraits.html
- for LTI systems, it may make sense to do the discrete → continuous conversion
- https://en.wikipedia.org/wiki/Logarithm_of_a_matrix#Calculating_the_logarithm_of_a_diagonalizable_matrix

If this does work, then:

- [ ] See how context input (and train std? reach directions?) shifts the plot

## Training

**See also [[results-training-methods-part2]].**

- [ ] Debug the equinox vmap warning that keeps showing up at the start of each training run (in 2-1 anyway)
- [ ] Parameter scaleup for part 2
- [ ] **Looks like `train_step` is re-compiling on the first batch, again, since I started passing `batch`. Debug.**
	- Interesting that in the baseline case, the “Training/validation step compiled in…” notices register as ~0 s for the condition (i.e. continuation of baseline) training run, which makes sense given that the identical functions were just compiled for the baseline run; however, there is still a ~5 s delay between when the tqdm bar appears, and when it starts moving!

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

## Database

- [x] Add more relevant properties to record models e.g. `ModelRecord`. For example, `where_train_strs` has to be stored in the database as a `dict[str, list[str]]`, but we might want to 1) load it as `dict[int, list[str]]` since the keys are actually indices of training iterations; or even 2) add a `where_train` property that automatically does the conversion when accessing the value
- [ ] Linking table that maintains foreign key relationships between evaluation records and model records; i.e. currently we store multiple model hashes in a `list[str]` column of the evals table, since an eval can depend on multiple models. But it doesn’t make sense to have an arbitrary number of foreign key columns in this table. Instead, the linking table would have a single entry for each eval-model dependency relation (i.e. a single eval record that refers to the hashes of 5 model records, corresponds to 5 records in the linking table).
- [ ] Function which deletes/archives any .eqx files for which the database record is missing – this will allow us to delete database records to “delete” models, then use this function to clean up afterwards
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
### Figures

- [ ] ~~Return records as a dataframe ~~
- [ ] **Optionally add annotations during retrieval**
- [ ] **Generate filenames/save to specified directory during retrieval**

#### Function to obtain figures based on training + testing conditions

Each figure type also has certain parameters which may vary. We may want a function that tells us which parameters are seen to vary for each figure subtype. This will help with writing more specific queries in the figure table, once we know what train+test conditions we are interested in.

### Motivation

It is kind of annoying to have to store figures in so many subdirectories, and to have to reconfigure the “suffixes” whenever I want to mess with a different hyperparameter. 

Instead, if we stored models and figures using (say) timestamps + hashes, and then saved them as entries in databases (one for models, one for figures?) along with all their hyperparameters, then we could easily filter/query and load/compare them based on those hyperparameters, rather than doing ugly and ungeneralizable parsing of filenames.

It might also make sense to automatically save evaluated states to disk, and to only re-evaluate them if the hash changes for the loaded model (i.e. because we have retrained the same set of hyperparameters, resulting in different trained weigths.) However, this might also be more trouble than it is worth.

## Figures and formatting

- [ ] Fix the yaxis range for measure plots in nb 1-2; some of them should display negative values
- [ ] Show trial, replicate, condition info in hoverinfo of individual *aligned* trajectories
- [ ] **Fix `add_endpoint_traces`**. Or was that feature part of the new trajectory plotter? If the latter, update notebooks 1-1 and 1-2 to use the feature
## Debris


- [ ] Double check that `save_model_and_add_record` is detecting existing records – not just generating a new hash and save a new model even if all the hps match
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

## Meetings

- [ ] Schedule a [[02 Questions#Steve|meeting]] with Steve. For one, ask about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)

