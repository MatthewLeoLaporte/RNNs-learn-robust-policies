---
created: 2024-10-04T11:27
updated: 2024-11-13T10:32
---
**See [[results-2|Part 2 results]] for ongoing analysis TODOs.**

- [ ] Parameter scaleup for part 2
- [ ] Add constant field aligned trajectory results to [[results-2]]
- [ ] New measure: covariance between network output and system noise variables
- [x] Use new `measure.py` in 1-2b
- [x] Compute measures in 2-2
- [ ] Distribution of direction of max net force
- [ ] **Merge `feature-database` into `main`**
- [ ] See how much more robust the baseline model is in part 1, if we **decrease the weight on the control cost**. (Talk to Gunnar about this – it will also affect the time urgency.)

## Analysis

- [ ] For each train std. in the “std” method, there is a certain context input at which the position profile is closest to a straight reach. Determine what this context input is. (I’m not sure about error bounds; I guess we would do the optimization for *all* validation trials individually.)

## Training

- [ ] Include the baseline condition in part 2, not for the aligned trajectory plots but for the measure plots
- [ ] Train on curl std 0.5, 1.0, 1.5
- [ ] Train with a small amount of noise (0.01?) in every case; the 

### Other technical stuff

- [ ] Looks like `train_step` is re-compiling on the first batch, again, since I started passing `batch`. Debug.
	- Interesting that in the baseline case, the “Training/validation step compiled in…” notices register as ~0 s for the condition (i.e. continuation of baseline) training run, which makes sense given that the identical functions were just compiled for the baseline run; however, there is still a ~5 s delay between when the tqdm bar appears, and when it starts moving!
- [ ] Try -1 and 1 for the “active” trianing variant, not 0 and 1?
	- I’m not sure this makes sense, since we want negative values of context input to be “anti-robust”
- [ ] Move part 1 training to a script + a yaml file defining which hyperparameters to train — or otherwise we’ll have to use batch quarto render 

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

- [ ] Show trial, replicate, condition info in hoverinfo of individual *aligned* trajectories

## Meetings

- [ ] Schedule a [[02 Questions#Steve|meeting]] with Steve. For one, ask about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)

