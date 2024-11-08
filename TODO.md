---
created: 2024-10-04T11:27
updated: 2024-11-08T11:39
---
- [x] [[#Part 1 results review and synthesis|Review part 1 results]]
- [x] Clean up this file
- [x] Normal distribution is notated $\mathcal{N}(\mu,\sigma^{2})$, not $\mathcal{N}(\mu,\sigma)$ – fix context annotations
- [x] Try a learning rate schedule
- [ ] Move part 1 training to a script + a yaml file defining which hyperparameters to train — or otherwise we’ll have to use batch quarto render 
- [x] Move post-training analysis (best replicates, excludes, loss plots, etc) to a script so that we can run it easily without needing to re-train models
- [ ] **Exclude models based on 1) # stds above the *best* replicate, and possibly 2) behavioural measures rather than total loss.** It looks like max net control force or end position error might be good indicators. 
- [ ] Schedule a [[02 Questions#Steve|meeting]] with Steve. For one, ask about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)
## Workflow

- [ ] Exclude from the replicate-comparison violins, any replicates which *for either the zero or the highest training condition* were excluded from analysis.
- [ ] Serialise `included_replicates` and `best_replicate` etc. in `post_training.py`, avoiding the need to cast the keys/values back to floats/arrays in a structure-dependent way, locally in each analysis notebook

## Formatting

- [ ] The context annotations are cut off for the velocity/force profile plots in 1-2b
- [ ] Fix x axis tick labels for random constant field train stds, for “performance_measures/compare_train_conditions”
- [ ] Violin figures could be much smaller overall
- [ ] Reduce width of “performance_measures/compare_train_conditions” in 1-2a
- [ ] Show trial, replicate, condition info in hoverinfo of individual *aligned* trajectories
- [ ] ~~Plot std bounds (or similar) for aligned 2D trajectories; plotting all the individual trials is too expensive once they are aligned. I’m not sure how to plot filled areas between fully 2D curves; instead it might make sense to use a [KDE](https://plotly.com/python/2d-histogram-contour/) with a single contour.~~ It’s hard to plot multiple KDEs on the same subplot in different colors. It would make more sense to plot confidence bounds, but this is also tricky. Leaving it be for now; instead I will downsample the curves if they exceed a specified quantity.


