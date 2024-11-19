---
created: 2024-10-04T11:27
updated: 2024-11-13T10:32
---

- [ ] **Debug NaN in ‘std’ variant** 
- [ ] **Goal steady-state fixed points**
- [ ] **Eigendecomposition of steady-state Jacobians**
- [ ] Move part 1 training to a script + a yaml file defining which hyperparameters to train — or otherwise we’ll have to use batch quarto render 
- [ ] Try -1 and 1 for the “active” trianing variant, not 0 and 1

## NaN in curl ‘std’ variant

Considering 100 example training trials from the respective tasks, with train std = 1.6

- The position and velocity losses are blowing up to huge values (~1e11) on the very first training iteration
- Context inputs are somewhat larger and higher-variance for the ‘std’ case, than the amplitude case (1.29&0.96 versus 0.80&0.60)
- For ‘amplitude’, the intervention `scale` is always 1.6, whereas for ‘std’ it is equal to the context input
- The intervention `amplitude` is identical for both ‘std’ and ’amplitude’; for ‘amplitude’ it is of course equivalent to the context input
- The intervention `active` is always `True` for both variants (because I have currently set it so)

None of this seems particularly critical. One possibility is that the slightly higher context inputs are driving the network into instability. If we get a typical batch (250 samples), what are the extremes of the context input?

- In the ‘amplitude’ case, the largest context input for a single sample of 250 trials is 2.7
- For ‘std’, it is 4.4

This could be significant. 

Here is the distribution of the context input over 25,000 trials of ‘amplitude’:

![[file-20241119160413412.png]]

And the same for ‘std’:

![[file-20241119160444596.png]]

These are both half-normal as expected, however ‘std’ is much wider. 

This is because the context input for `amplitude` is just a standard normal sample, which is then multiplied by 1.6 to get the actual field strength; 

whereas the context input for `std` is 1.6 *times* a standard normal sample, which is then multiplied by another standard normal sample to get the field strength. It may make more sense to pass a standard normal sample in both cases, and keep the scaling by `field_std` in whichever part of the intervention params is *not* passed as context input. Thus the context input would behave essentially the same although its relationship to the actual field strength would change.

Comparing to the constant field case, where the ‘std’ variant trained just fine even though `field_std` was multiplied by the normal sample prior to assigning the context input… the `field_std` is just smaller, so while the distribution for ‘std’ is still wider than for ‘amplitude’, overall it is no wider than it is for the ‘amplitude’ variant in the curl case. So the absolute amplitude of the context input may still matter

**I tried this out and it looks like it works… was able to train for 500 iterations for all three methods, without NaN resulting**



## Model loss terms

- [x] Construct `history.loss_validation` from `task.loss_func` only, when a custom loss is passed to `TaskTrainer`; thus we don’t need to associate model loss terms with the task instance, but can pass the customized loss directly to `TaskTrainer` as I wanted
- [ ] That also means we don’t need to know about the custom loss alterations when calling `get_task_model_pairs`. However, we still need a deserialisation function for `TaskTrainerHistories`… maybe `TaskTrainer` should generally be able to return this?

## Database

- [ ] ~~Maybe: Give post-training models their own table, with a reference to the `model_hash` they originated from; then in `post_training`, optionally skip post-training of models when there is already a record with matching `model_hash` and post-training hyperparameters.~~
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

- [ ] Add zero hline on velocity plots
	- I already did this in 1-2a, but I think it was not visible. Moved to `fbp.profiles` so that it is plotted before the profiles themselves
- [ ] ~~The context annotations are cut off for the velocity/force profile plots in 1-2b~~
- [ ] Show trial, replicate, condition info in hoverinfo of individual *aligned* trajectories
- [ ] ~~Plot std bounds (or similar) for aligned 2D trajectories; plotting all the individual trials is too expensive once they are aligned. I’m not sure how to plot filled areas between fully 2D curves; instead it might make sense to use a [KDE](https://plotly.com/python/2d-histogram-contour/) with a single contour.~~ It’s hard to plot multiple KDEs on the same subplot in different colors. It would make more sense to plot confidence bounds, but this is also tricky. Leaving it be for now; instead I will downsample the curves if they exceed a specified quantity.

## Meetings

- [ ] Schedule a [[02 Questions#Steve|meeting]] with Steve. For one, ask about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)

## Bigger things

- [ ] Vmap task generation, then train in parallel? i.e. instead of using a loop and getting a `TrainStdDict`, we would add another batch dimension to the model parameter arrays. The `trainer` would need to vmap separately over the task and over the ensemble