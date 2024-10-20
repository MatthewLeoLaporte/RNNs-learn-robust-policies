---
created: 2024-10-04T11:27
updated: 2024-10-20T14:41
---
- [ ] Modify nb 1-2a to work with both disturbance types – mostly just renaming `curl_*` to `disturbance_*`, but also need to decide on file naming convention (in particular when train disturbance is not the same as test disturbance)
- [ ] **Quantify response to feedback perturbation – max leftward and rightward control force** 
- [ ] **Plot velocity profiles for feedback perturbations**
- [ ] Add notebook to load models for multiple noise conditions, and plot distribution comparisons 
- [x] Add endpoints to 1-2a 1.7.2 plots (single-condition trajectories across curl amplitude)
- [ ] Try using a format string with a slash in `render_params.yaml` to store HTML output in sub-subdirectories according to  evaluation params (similarly to how the figures are stored)
- [x] **Choose better colorscales for train and test curl amplitude comparisons**
- [ ] Max forward velocity – comparing different peaks…
- [ ] Ask Steve about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)
## Review

- [x] Ensure `n=1` for no-noise case.
- [x] Doesn’t make sense to change delay between train and test.
- [x] Don’t need to show single-condition distributions – it’s enough to see that the error bars are reasonably small on the example trajectories
- [x] Use different colors for 2x2 low-high summary plots, and replicate comparison plots (curl amplitude in one, train curl std. in the other)
- [x] Use “Train curl std.” with a float for the replicate comparison plots, not “No curl during training” etc.

### Exclude model replicates whose training diverged

Or else save/load earlier checkpoint. 

This only appears to be necessary in the case of training with delay, but we can implement it more generally. 

It might be easier to do this at save time. 

Aside from the loss, it looks like max net control force or end position error might be good indicators. 
![[file-20241018152818864.png]]
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


