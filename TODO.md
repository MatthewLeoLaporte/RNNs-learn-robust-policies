---
created: 2024-10-04T11:27
updated: 2024-11-13T10:32
---

- [x] Something is wrong with the model replicate randomization, maybe; looking at the distribution plots comparing replicates, 3 have identical distributions, as do the other 2.
- [x] Open `.qmd` files as .py and remove unnecessary imports
- [x] **Train 1-1 zero noise while fixing readout**
- [x] **Add readout norm plot to post_training**
- [ ] **Debug NaN in ‘std’ variant** 
- [ ] **Goal steady-state fixed points**
- [ ] **Eigendecomposition of steady-state Jacobians**
- [ ] Move part 1 training to a script + a yaml file defining which hyperparameters to train — or otherwise we’ll have to use batch quarto render 
- [x] Retrain part 1 models… again
	- Make sure to get rid of `MASS` from `setup_task_model_pairs` first
	- Also training different network sizes, now that 

## Troubleshooting model replicates

- Their model weights aren’t identical
- Ah, it’s just because I was passing a boolean index array when I should have been passing integers for `included_replicates` – so it was just selecting replicate 0 (False) or 1 (True) repeatedly

## Workflow

### Database

- [ ] ~~Include `disturbance_type_train` and `disturbance_type` in figure records, even though this is redundant with the models table~~
- [ ] ~~It looks like sometimes, running 1-1, if we change certain parameters (e.g. `n_batches`) then we will end up overwriting the models file (hash is the same? how can that be?) but not the train_history file. Double check that this is actually happening. If so, there’s an issue with checking and deleting previous runs, and some hyperparameters are not being accounted for when deciding to delete.~~
- [ ] Automatically export CSV or something for each db table, as a backup

#### Figures

- [x] Can remove `add_context_annotation` calls throughout notebooks, and replace them with an option to annotate a figure when retrieving it from the db
- [ ] The figure retrieval function should also allow us to define a filename format in terms of available columns

##### Function to obtain figures based on training + testing conditions

Each figure type also has certain parameters which may vary. We may want a function that tells us which parameters are seen to vary for each figure subtype. This will help with writing more specific queries in the figure table, once we know what train+test conditions we are interested in.

##### 

#### Motivation

It is kind of annoying to have to store figures in so many subdirectories, and to have to reconfigure the “suffixes” whenever I want to mess with a different hyperparameter. 

Instead, if we stored models and figures using (say) timestamps + hashes, and then saved them as entries in databases (one for models, one for figures?) along with all their hyperparameters, then we could easily filter/query and load/compare them based on those hyperparameters, rather than doing ugly and ungeneralizable parsing of filenames.

It might also make sense to automatically save evaluated states to disk, and to only re-evaluate them if the hash changes for the loaded model (i.e. because we have retrained the same set of hyperparameters, resulting in different trained weigths.) However, this might also be more trouble than it is worth.

## Formatting

- [ ] The context annotations are cut off for the velocity/force profile plots in 1-2b
- [ ] Show trial, replicate, condition info in hoverinfo of individual *aligned* trajectories
- [ ] ~~Plot std bounds (or similar) for aligned 2D trajectories; plotting all the individual trials is too expensive once they are aligned. I’m not sure how to plot filled areas between fully 2D curves; instead it might make sense to use a [KDE](https://plotly.com/python/2d-histogram-contour/) with a single contour.~~ It’s hard to plot multiple KDEs on the same subplot in different colors. It would make more sense to plot confidence bounds, but this is also tricky. Leaving it be for now; instead I will downsample the curves if they exceed a specified quantity.

## Meetings

- [ ] Schedule a [[02 Questions#Steve|meeting]] with Steve. For one, ask about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)

## Bigger things

- [ ] Vmap task generation, then train in parallel? i.e. instead of using a loop and getting a `TrainStdDict`, we would add another batch dimension to the model parameter arrays. The `trainer` would need to vmap separately over the task and over the ensemble