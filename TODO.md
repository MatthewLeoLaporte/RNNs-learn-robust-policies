---
created: 2024-10-04T11:27
updated: 2024-11-13T10:32
---

- [x] Debug NaN in ‘std’ variant
- [ ] **Goal steady-state fixed points**
- [ ] **Eigendecomposition of steady-state Jacobians**
- [ ] Move part 1 training to a script + a yaml file defining which hyperparameters to train — or otherwise we’ll have to use batch quarto render 
- [ ] Try -1 and 1 for the “active” trianing variant, not 0 and 1?
	- I’m not sure this makes sense, since we want negative values of context input to be “anti-robust”

## Training

- [ ] Reset `opt_state` at iteration 500
- [ ] Pre-training for 1000 steps without perturbations
- [ ] Ramping up perturbations – the only way I see to do this atm is to modify task specs to have the signature `batch, key` rather than just `key`…
- [ ] Looks like `train_step` is re-compiling on the first batch, again, since I started passing `batch`. Debug.

## Model loss terms

- [x] Construct `history.loss_validation` from `task.loss_func` only, when a custom loss is passed to `TaskTrainer`; thus we don’t need to associate model loss terms with the task instance, but can pass the customized loss directly to `TaskTrainer` as I wanted
- [x] ~~That also means we don’t need to know about the custom loss alterations when calling `get_task_model_pairs`. However, we still need a deserialisation function for `TaskTrainerHistories`… maybe `TaskTrainer` should generally be able to return this?~~

## Database

- [ ] Function which deletes/archives any .eqx files for which the database record is missing – this will allow us to delete database records to “delete” models, then use this function to clean up afterwards
- [ ] Automatically export CSV or something for each db table, as a backup

### Figures

- [ ] Return records as a dataframe 
- [ ] **Add annotations during retrieval**
- [ ] **Generate filenames/save to specified directory during retrieval**

#### Function to obtain figures based on training + testing conditions

Each figure type also has certain parameters which may vary. We may want a function that tells us which parameters are seen to vary for each figure subtype. This will help with writing more specific queries in the figure table, once we know what train+test conditions we are interested in.

### Motivation

It is kind of annoying to have to store figures in so many subdirectories, and to have to reconfigure the “suffixes” whenever I want to mess with a different hyperparameter. 

Instead, if we stored models and figures using (say) timestamps + hashes, and then saved them as entries in databases (one for models, one for figures?) along with all their hyperparameters, then we could easily filter/query and load/compare them based on those hyperparameters, rather than doing ugly and ungeneralizable parsing of filenames.

It might also make sense to automatically save evaluated states to disk, and to only re-evaluate them if the hash changes for the loaded model (i.e. because we have retrained the same set of hyperparameters, resulting in different trained weigths.) However, this might also be more trouble than it is worth.

## Formatting

- [x] Add zero hline on velocity plots
	- I already did this in 1-2a, but I think it was not visible. Moved to `fbp.profiles` so that it is plotted before the profiles themselves
- [ ] Show trial, replicate, condition info in hoverinfo of individual *aligned* trajectories

## Meetings

- [ ] Schedule a [[02 Questions#Steve|meeting]] with Steve. For one, ask about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)

## Bigger things

- [ ] Vmap task generation, then train in parallel? i.e. instead of using a loop and getting a `TrainStdDict`, we would add another batch dimension to the model parameter arrays. The `trainer` would need to vmap separately over the task and over the ensemble