---
created: 2024-10-04T11:27
updated: 2024-11-07T23:11
---
- [x] [[#Part 1 results review and synthesis|Review part 1 results]]
- [ ] **Clean up this file**
- [x] Normal distribution is notated $\mathcal{N}(\mu,\sigma^{2})$, not $\mathcal{N}(\mu,\sigma)$ – fix context annotations
- [ ] **Try a learning rate schedule when training part 2**
- [ ] **Move part 1 training to a script + a yaml file defining which hyperparameters to train** — or otherwise we’ll have to use batch quarto render 
- [x] Move post-training analysis (best replicates, excludes, loss plots, etc) to a script so that we can run it easily without needing to re-train models
- [ ] **Exclude models based on 1) # stds above the *best* replicate, and possibly 2) behavioural measures rather than total loss.** It looks like max net control force or end position error might be good indicators. 
- [ ] Schedule a [[02 Questions#Steve|meeting]] with Steve. For one, ask about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)
## Part 1 results review and synthesis
### Important notes

- There are 3 train and 3 test perturbation conditions (control, curl, random) such that for each noise+delay condition we can do 3x3 train-test comparisons
- There are 3 delay conditions (0, 2, 4 steps); these do not vary between 
- There are 3 noise conditions (0, 0.04, 0.1)

### TODO

- [ ] Serialise `included_replicates` and `best_replicate` etc. in `post_training.py`, avoiding the need to cast the keys/values back to floats/arrays in a structure-dependent way, locally in each analysis notebook
- [ ] The context annotations are cut off for the velocity/force profile plots in 1-2b
- [ ] Fix x axis tick labels for random constant field train stds, for “performance_measures/compare_train_conditions”
- [ ] Reduce width of “performance_measures/compare_train_conditions” in 1-2a
- [ ] Show trial, replicate, condition info in hoverinfo of individual *aligned* trajectories
- [ ] ~~Plot std bounds (or similar) for aligned 2D trajectories; plotting all the individual trials is too expensive once they are aligned. I’m not sure how to plot filled areas between fully 2D curves; instead it might make sense to use a [KDE](https://plotly.com/python/2d-histogram-contour/) with a single contour.~~ It’s hard to plot multiple KDEs on the same subplot in different colors. It would make more sense to plot confidence bounds, but this is also tricky. Leaving it be for now; instead I will downsample the curves if they exceed a specified quantity.
- [ ] Exclude from the replicate-comparison violins, any replicates which *for either the zero or the highest training condition* were excluded from analysis.

## Gunnar meeting

### Comparison of types of robustness

Observations:

- In the presence of a constant random field, the network must output a constant non-zero force to remain stationary at the goal. The models are able to do this, regardless of whether they were trained on random fields; however, the control models do a ~straight reach to a position that is rotated away from the goal, almost like the first (naive) trial of a visuomotor rotation task,
- The forward velocity profiles are *identical* in the presence and absence of a random field, for all models, regardless of what perturbation the model was trained on 
- However, there is a difference in certain related measures (max net force?) between the model trained in the presence of random fields, versus not.
- Training on random fields initially leads to a little “hook” correction at the end of the reach, in addition to a reduction in the slope of deviation during the rest of the reach. At higher train std, a smoother curvature of the solution is achieved.
- Compensation for random fields is much less sensitive to delays, versus curl fields. This makes sense since there isn’t feedback between control forces and orthogonal velocities. 
- Likewise, networks trained on curl fields + delays tend to be worse at all tasks, presumably because it was harder for them to learn any coherent policy to reach the goal.

When we switch the disturbance type during testing:

- Training on curl reduces deviations for random fields, but does not totally eliminate endpoint error
- Training on random fields reduces deviations in the presence of modest curls, but also leads to oscillations around the goal for larger curls

How can we interpret this?

- Should we train on a combination of the two? 

In part 2 of the project we may be able to interpret these differences more easily.

#### Is there another kind of disturbance that we should try? 

##### **Random velocity-dep**

- this includes curl fields as a special case 
- curl fields may be the most interesting case of a velocity dependent field because of their orthogonality
- whereas a velocity dependent field parallel to the velocity vector would merely lead to changes in gain/limitations on achievable accelerations

##### Force transformation

Acceleration/force – e.g. a task rotation, forces on the point mass actually are applied partially in another direction, etc.

### Choice of feedback impulse amplitudes

- How to make pos vs. vel perturbations comparable?
- Choose amplitudes to align the max (or sum?) deviation for the control (zero train std) condition?

### Choice of disturbance train stds 

- Levels are different for random and curl
- I think it’s mostly fine so long as we can show a spread of robustness behaviour for each disturbance type. 
- The test amplitude for random fields could be a bit higher.

### Exclusion of replicates from analysis

Currently based on standard deviations away from the mean best loss.

### Confidence bounds in parametric trajectory plots

Currently just plotting a sample of individual curves; but this can be messy.

### Velocity peaks 

Max forward velocity – quantify number/amplitude of peaks? 


## Motivation and questions

What are the emergent properties of a system that learns to be robust? 

That is, what is the effect of model uncertainty during training, on the learned network policies for motor control?

> [!NOTE]+
> **No Hebbian reinforcement** in this project!

## Results: Part 1

### Preliminary stuff

- Maybe show adaptation (i.e. if we train only on CW curl fields, networks adapt)

### Training on different levels of model uncertainty

In the simplest case:

- one ensemble with some level of model uncertainty (e.g. fine tuned on balanced curl fields with some std)
- one ensemble without model uncertainty (only baseline system noise)

#### Optimization of convergence

It may be necessary to do one or more of the following to get optimal models:

- Introduce perturbations after initial training period without them
- Introduce delay (gradually) after initial training period without it
- Learning rate schedule
- Batch size schedule (increase later in training)
- Gradient clipping
- Use `optax.adamw` for weight decay regularization

### Perturbation analysis

#### Model perturbation analysis

Demonstrate that in general behavioural terms, the models trained on perturbations are actually more robust. 

- [x] Perturb the two ensembles with constant curl fields
- [x] Show examples of single-condition differences 
- [x] Examine performance measures: endpoint error, max/sum deviations
	- Others: time to target, max/sum control forces
	- Compare variance across trials vs. across replicates

Also: 

- Examine the relationship between field strength and performance measures (e.g. does endpoint error decrease more slowly for robust network, as curl field strength increases?)

#### Feedback perturbation analysis

Demonstrate that the robust model is more reactive to feedback perturbations (at steady state).

- Plot of single/average trials comparing the output response to a feedback impulse
- Plot of maximum output magnitude versus feedback impulse magnitude

### Network analysis

e.g. change in unit gains

This will largely be left to part 2, since it is difficult to perform on networks that were trained/fine-tuned separately. It might make sense to perform a supplementary analysis to demonstrate this difficulty. 

### Additional analyses

#### Different kinds of perturbations

It is important to demonstrate that the results

For example, comparison between state-dependent/”closed-loop” perturbations (e.g. curl fields) versus state-independent/”open-loop” perturbations (e.g. random constant fields). Or perhaps just repeat the entire analysis for fixed fields. 

Also train/fine-tune a network on increased system noise and show whether it induces any robustness – presumably not, since symmetric noise will not cause local system deviations that the network would interpret as model uncertainty.

## Results: Part 2

### Training a single network on different levels of model uncertainty

- i.e. a single network that is trained on both perturbations and no perturbations
- The network has 1 additional scalar input that tells whether it is being trained in a perturbed context (1) or not (0)

### Perturbation analysis

- Show that in a perturbed context, endpoint error decreases as we increase the context input from 0 to 1. 
- Show that reactivity to feedback perturbations increases with context input. 

### Network analysis

Evaluate trials with different values of the context input, and compare:

- Gains of individual units
- Dimensionality reduction: the overall dynamical structure (e.g. rotations?) of the trial states. Plot some streamlines. 
- Fixed points; e.g. how do goal-goal manifolds interpolate with the context input?

Can we easily compare unit gains with dynamical changes (e.g. steepening around goal-goal attractors)?

## Methods

### Statistics

- Across trials
- Across networks
 
It should be OK to just show distributions, without needing to quantify significance

### Hyperparameters

Supplementary analyses showing that network size, activation function, etc. are not critical to results.

## Discussion 

### Robust control theory

- Feedback reactivity

We have not made direct comparisons in this study. Future studies would separate the feedback weights etc, or possibly have a separate state estimation network. 

### Network analysis

### Limitations etc.
#### Separation principle

We use undifferentiated networks. 

However, certain things are harder to investigate in this context.

- What does the network’s forward model look like? For example, in the case of adaptation (CW curl) vs. robustening (mixed direction curl) what is the difference in the effect on the forward model?

Note that in the future we can approach this problem without needing entirely distinct networks. For example, weight partitioning.


