---
created: 2024-10-04T11:27
updated: 2024-11-13T10:32
---
**See [[results-2|Part 2 results]] for ongoing analysis TODOs.

- [ ] **Poincare graph - phase portraits** (see work scratch)
	- http://phaseportrait.blogspot.com/2013/01/discrete-time-phase-portraits.html
	- for LTI systems, it may make sense to do the discrete → continuous conversion
	- https://en.wikipedia.org/wiki/Logarithm_of_a_matrix#Calculating_the_logarithm_of_a_diagonalizable_matrix
- [ ] **Try a leaky vanilla RNN**
- [ ] See how much more robust the baseline model is in part 1, if we **decrease the weight on the control cost**. (Will also affect the time urgency.)
	- My expectation is that it will of course increase the forces and decrease the movement duration, but that depending on the perturbation this will actually make it *less* robust (e.g. curl fields)
- [ ] Send inputs to only a subset of units; readout from only a subset
	- For the inputs, I think: store a list (array) of matrices describing the mapping for each index in the input
	- A distribution in space (e.g. arrange them in a grid and use a distance measure) is fine
	- However, that may be unnecessary complexity; if we can easily specify the joint distribution of input connections 
	- e.g. there are 3 types of input connections, each corresponding to a different subset/slice of the input array 

## Efficiency

- [ ] Maybe save a separate model table record in the database, for each *separate* disturbance std during training. This will:
	- Prevent us from saving the same trained model multiple times because e.g. once we trained three runs `[0, 0.1, 1]` and another time two runs `[0, 0.1]`.
	- Force us to use the multi-loading logic in the analyses notebooks (i.e. load db entry with std 0, and db entry with std 0.1, and compare); note that this kind of logic will be necessary anyway for some of the supplementary analyses (e.g. noise comparisons).
	- Note that the foreign refs in the eval and fig tables would change into sets/lists, which is probably as it should be.
- [ ] Convert all the notebooks into scripts
- [x] Make wrapper for `go.Figure` that adds a “copy as png” and “copy as svg” buttons; this will save a lot of time
	- See https://plotly.com/python/figurewidget-app/


## Technical

- [ ] Write a `tree_stack` that works with models – can’t just use `apply_to_filtered_leaves` since the shape of the output is changed wrt the input 
- [ ] Try vmapping over `schedule_intervenor`, to get batch dimensions in models and tasks. Batched tasks seems like a missing link to making some of the analyses easier to write.

## Analysis

### Fixed points

- [ ] Hessians - describe the curvature
- [ ] Examine reaching FP trajectories for both baseline (no disturbance) and disturbance conditions

#### Troubleshooting

- [ ] **Examine the loss curve over iterations of the FP finding optimization**; possibly, plot the **hidden state over the optimization**; will hopefully give an idea what is different about the conditions that fail to converge
- [ ] Try to get rid of dislocations in fixed point trajectories (though they aren’t very bad except at context -2)

#### Translation invariance

- [x] Do fixed points vary with goal position (i.e. target input), or just with the difference between target and feedback inputs?
	- for example, do the fixed points change if we translate the target and feedback position inputs in the same way?
	- The “ring” of steady-state (“goal-goal”) FPs for simple reaching suggests this might be the case. 
	- **They do vary. The grid of steady-state FPS form a planar grid** in the space of the top three PCs. As context input changes, this grid translates along an ~orthogonal direction. **Should plot the readout vector** and see if the direction of translation is roughly aligned with it, which would make sense. 
- [ ] ~~(In other words?:) Is there only a single steady-state fixed point for each trained network (+context input)?~~
	- If not, do all steady-state FPs show the same kinds of behaviour (e.g. increasing context input → more contracting?)
	- **Based on the eigenspectra**, the steady-state grid FPs are approximately similar in their dynamical properties. There appears to be significantly more variation between context inputs, than between positions. If we zoom in on a particular eigenvalue, it appears to shift its position in tiny ~grids reflecting the change in position.

#### Network input perturbations

i.e. find steady-state FPs, then change the feedback input and see if the location/properties of the FP changes

- [ ] As we increase the perturbation size, do the eigenvectors become aligned? Do they point in the direction in state space that would increase the readout in the corrective direction?
- [ ] **Identify the most unstable eigenvectors. Which units have the strongest contribution?** If we perturb these units at steady state, versus some other randomly selected set of units, do we get a large “corrective” change in the network output?

#### Hessian analysis

- Demonstrate that the fixed points are dynamical minima (vs. say saddle-points)? (Positive definite?)
- Does the curvature change with training std? Context input?

#### FP trajectories

How do the fixed points change over reach/stabilization trajectories? 

- [ ] **How do the init-goal and goal-goal fixed point structures vary with context input?**
- [ ] **What about the init-goal fixed points for a feedback perturbation (i.e. if we suddenly change the network input)?**

#### Other questions

- [ ] What are the “wing” eigenvalues, e.g. seen for the origin steady-state FP, in DAI+curl?
	- Note that they seem to be the strongest oscillatory modes
	- They become larger (i.e. decay more slowly) and higher-frequency with increasing context input 
	- They become relatively larger and higher-frequency when training on stronger field std.

### Measures

- [ ] For each train std. in the “std” method, there is a certain **context input at which the position profile is closest to a straight reach**. Determine what this context input is. (I’m not sure about error bounds; I guess we would do the optimization for *all* validation trials individually.)
- [ ] **Direction of max net force (at start of trial?)**

## Training

**See also [[results-training-methods-part2]].**

- [ ] Debug the equinox vmap warning that keeps showing up at the start of each training run (in 2-1 anyway)

### Best training method 

Out of [[methods#Training methods|these]]. This is unclear. 

- It looks like the ideal set of field stds. to get a good spread of measures/robustness varies between methods
- [x] Go through the sets of trained models I’ve made recently and find the “best spread” for each method

#### Current state

- BCS is in some ways the best for constant fields, as behaviour is approximately the same across training stds, for context input 0
- BCS does not even work for curl fields
- DAI works in both cases, but the results show the worst spread of robust behaviour, probably because the network has little uncertainty about how perturbed it will be
- PAI works in both cases, but appears to be [[2024-12-13#PhD|inducing]] a bias/adaptation in the curl field case.

### Other technical stuff

- [ ] Parameter scaleup for part 2
- [ ] **Looks like `train_step` is re-compiling on the first batch, again, since I started passing `batch`. Debug.**
	- Interesting that in the baseline case, the “Training/validation step compiled in…” notices register as ~0 s for the condition (i.e. continuation of baseline) training run, which makes sense given that the identical functions were just compiled for the baseline run; however, there is still a ~5 s delay between when the tqdm bar appears, and when it starts moving!
- [ ] ~~Try -1 and 1 for the “active” trianing variant, not 0 and 1?~~
	- I’m not sure this makes sense, since we want negative values of context input to be “anti-robust”
- [ ] Move part 1 training to a script + a yaml file defining which hyperparameters to train — or otherwise we’ll have to use batch quarto render 

## Database

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

- [ ] Return records as a dataframe 
- [ ] **Optionally add annotations during retrieval**
- [ ] **Generate filenames/save to specified directory during retrieval**

#### Function to obtain figures based on training + testing conditions

Each figure type also has certain parameters which may vary. We may want a function that tells us which parameters are seen to vary for each figure subtype. This will help with writing more specific queries in the figure table, once we know what train+test conditions we are interested in.

### Motivation

It is kind of annoying to have to store figures in so many subdirectories, and to have to reconfigure the “suffixes” whenever I want to mess with a different hyperparameter. 

Instead, if we stored models and figures using (say) timestamps + hashes, and then saved them as entries in databases (one for models, one for figures?) along with all their hyperparameters, then we could easily filter/query and load/compare them based on those hyperparameters, rather than doing ugly and ungeneralizable parsing of filenames.

It might also make sense to automatically save evaluated states to disk, and to only re-evaluate them if the hash changes for the loaded model (i.e. because we have retrained the same set of hyperparameters, resulting in different trained weigths.) However, this might also be more trouble than it is worth.

## Formatting

- [ ] Show trial, replicate, condition info in hoverinfo of individual *aligned* trajectories
- [ ] **Fix `add_endpoint_traces`**. Or was that feature part of the new trajectory plotter? If the latter, update notebooks 1-1 and 1-2 to use the feature

## Meetings

- [ ] Schedule a [[02 Questions#Steve|meeting]] with Steve. For one, ask about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)

