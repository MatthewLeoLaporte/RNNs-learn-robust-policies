---
created: 2024-10-04T11:27
updated: 2024-11-13T10:32
---
**See [[results-2|Part 2 results]] for ongoing analysis TODOs.

- [ ] Add figures to database in 2-6
- [x] Generate train std 0.0 aligned trajectory plots in 2-3 (as well as std 0.4/1.0 already)
- [x] Show train std 0.0 violins in the background (e.g. as semitransparent grey) of measure plots in 2-3
	- I’m not sure how to have some violins overlayed while others are grouped, so for now I’ve just plotted them side-by-side. 
	- This makes things rather skinny but when publishing we could use `get_underlay_fig` to produce a background layer and then composite them as images
- [ ] See how much more robust the baseline model is in part 1, if we **decrease the weight on the control cost**. (Will also affect the time urgency.)
	- My expectation is that it will of course increase the forces and decrease the movement duration, but that depending on the perturbation this will actually make it *less* robust (e.g. curl fields)
- [ ] Move training and analysis notebooks into separate directories – better repo organization
	- Make sure to find-replace
	- update how the database/IO works, if necessary
	- 0-training, 1-analysis, 2-supplement
	- What about `parameters/`?

## Efficiency

- [x] Make wrapper for `go.Figure` that adds a “copy as png” and “copy as svg” buttons; this will save a lot of time
	- See https://plotly.com/python/figurewidget-app/
- [ ] Automatically generate a list of figure types in all notebooks

## Technical

- [ ] Write a `tree_stack` that works with models – can’t just use `apply_to_filtered_leaves` since the shape of the output is changed wrt the input 
- [ ] Try vmapping over `schedule_intervenor`, to get batch dimensions in models and tasks. Batched tasks seems like a missing link to making some of the analyses easier to write.

## Analysis

### Fixed points

- [ ] Examine reaching FP trajectories for both baseline (no disturbance) and disturbance conditions
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

- Demonstrate that the fixed points are dynamical minima (vs. say saddle-points)
	- This could also help to reveal what’s going on a 

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
- [ ] Try -1 and 1 for the “active” trianing variant, not 0 and 1?
	- I’m not sure this makes sense, since we want negative values of context input to be “anti-robust”
- [ ] Move part 1 training to a script + a yaml file defining which hyperparameters to train — or otherwise we’ll have to use batch quarto render 

## Database

- [ ] **Function which deletes/archives any .eqx files for which the database record is missing** – this will allow us to delete database records to “delete” models, then use this function to clean up afterwards

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

