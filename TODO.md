---
created: 2024-10-04T11:27
updated: 2024-11-10T13:28
---

- [ ] Move part 1 training to a script + a yaml file defining which hyperparameters to train — or otherwise we’ll have to use batch quarto render 
- [ ] Retrain part 1 models… again
	- Make sure to get rid of `MASS` from `setup_task_model_pairs` first
- [x] **Exclude models based on 1) # stds above the *best* replicate, and possibly 2) behavioural measures rather than total loss.** It looks like max net control force or end position error might be good indicators. 
## Workflow

- [x] Exclude from the replicate-comparison violins, any replicates which *for either the zero or the highest training condition* were excluded from analysis.
- [x] Serialise `included_replicates` and `best_replicate` etc. in `post_training.py`, avoiding the need to cast the keys/values back to floats/arrays in a structure-dependent way, locally in each analysis notebook

### Database

It is kind of annoying to have to store figures in so many subdirectories, and to have to reconfigure the “suffixes” whenever I want to mess with a different hyperparameter. 

Instead, if we stored models and figures using (say) timestamps + hashes, and then saved them as entries in databases (one for models, one for figures?) along with all their hyperparameters, then we could easily filter/query and load/compare them based on those hyperparameters, rather than doing ugly and ungeneralizable parsing of filenames.

It might also make sense to automatically save evaluated states to disk, and to only re-evaluate them if the hash changes for the loaded model (i.e. because we have retrained the same set of hyperparameters, resulting in different trained weigths.) However, this might also be more trouble than it is worth.

## Formatting

- [ ] The context annotations are cut off for the velocity/force profile plots in 1-2b
- [ ] Show trial, replicate, condition info in hoverinfo of individual *aligned* trajectories
- [ ] ~~Plot std bounds (or similar) for aligned 2D trajectories; plotting all the individual trials is too expensive once they are aligned. I’m not sure how to plot filled areas between fully 2D curves; instead it might make sense to use a [KDE](https://plotly.com/python/2d-histogram-contour/) with a single contour.~~ It’s hard to plot multiple KDEs on the same subplot in different colors. It would make more sense to plot confidence bounds, but this is also tricky. Leaving it be for now; instead I will downsample the curves if they exceed a specified quantity.

## Meetings

- [ ] Schedule a [[02 Questions#Steve|meeting]] with Steve. For one, ask about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)