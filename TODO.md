---
created: 2024-10-04T11:27
updated: 2024-10-30T21:09
---
- [ ] **Review the results so far and make a summary of the ones that should appear as main/supplementary figures in paper**. Probably, make 2-3 files: 1) results that will almost certainly appear in the paper, 2) (maybe) results that may or may not be supplementary, 3) results that will be supplementary if they are included at all.
- [ ] **Move part 1 training to a script + a yaml file defining which hyperparameters to train** — or otherwise we’ll have to use batch quarto render to 
- [ ] **Move post-training analysis (best replicates, excludes, loss plots, etc) to a script so that we can run it easily without needing to re-train models**
- [ ] **Exclude models based on behavioural measures rather than total loss.** It looks like max net control force or end position error might be good indicators. 
- [ ] After finishing up analyses in 1-2b, re-run batch quarto render for both 1-2a and 1-2b, for all noise and delay conditions, as well as for both curl and random training disturbances. Maybe reduce to 3 (lo-mid-high) disturbance levels. Also try training disturbance stds up to 1.5 or so (to see if the secondary peak disappears from profile)
- [ ] Try a 200-step solution for networks trained on random and tested on curl, to see how the “goal orbit” evolves
- [ ] Add notebook to load models for multiple noise/delay conditions, and plot distribution comparisons (only do this after deciding which plots to make – it’s too complicated to run *all* the existing plots as noise comparisons)
- [ ] Max forward velocity – quantify number/amplitude of peaks…
- [ ] Schedule a [[02 Questions#Steve|meeting]] with Steve. For one, ask about sensory perturbations in human tasks – do they see oscillations (i.e. going from straight to “loopy”, like we see in the control vs. robust networks)
## Gunnar meeting

### Choice of disturbance train stds 

- Levels are different for random and curl
- I think it’s mostly fine so long as we can show a spread of robustness behaviour for each disturbance type. 

### Choice of feedback impulse amplitudes

- How to make pos vs. vel perturbations comparable?
- Choose amplitudes to align the max (or sum?) deviation for the control (zero train std) condition?

### Comparison of types of robustness

Basic observations:

- In the presence of a constant random field, the network must output a constant non-zero force to remain stationary at the goal. The models are able to do this, regardless of whether they were trained on random fields; however, the control models do a straight reach to a position that is rotated away from the goal, almost like the first trial of a visuomotor adaptation task
- The acceleration phase of the reach, as well as the max net force/velocity, are *identical* in the presence and absence of a random field, for all models, regardless of whether the model was trained in the presence of random fields (however, there is a difference in these measures between the model trained in the presence of random fields, versus not)
- Training on random fields initially leads to a little “hook” correction at the end of the reach, in addition to a reduction in the slope of deviation during the rest of the reach. At higher train std, a smoother curvature of the solution is achieved.
- Compensation for random fields is less sensitive to delays. This makes sense since there isn’t closed-loop coupling between control forces and orthogonal velocities. (How do networks trained on curl+delay perform, on random fields? and so on)

When we switch the disturbance type during testing:

- Training on curl reduces deviations for random fields
- Training on random fields reduces deviations *during* the reach in the presence of curl, but also leads to oscillations around the goal

How can we interpret this?

- Should we train on a combination of the two? 

**Presumably, in part 2 of the project we may be able to interpret these differences in terms of gains.**

#### Is there another kind of disturbance that we should try? 

- Acceleration/force – e.g. a task rotation, forces on the point mass actually are applied partially in another direction, etc.
- Parallel velocity-dep – like a curl field but parallel instead of orthogonal, such that there is positive feedback between velocity and acceleration in any direction

### Local disturbances

e.g. only apply the force field on a certain part of the reach

This might help to make conclusions about the “types” of robustness, and the nature of the network policies. 

However, I am not sure yet how to think about this. Worth discussing.

## Enforce rotational invariance of the learned strategy

This is motivated by the slightly different responses of the network when a feedback perturbation is in different directions. Is it possible to train the network so that its response in the x direction is just like a rotated version of its response in the y direction?

Some ideas:

- Provide the RNN inputs in a polar representation
- Make the RNN output the force vector in a polar representation, by forcing their trigonometric conversion to x/y 
- Add an explicit term to loss function; e.g. on each batch, evaluate on a big center-out set, and penalize for the difference between the control vectors when they are all rotated to lie in the same reach direction
- Modify the network architecture to enforce symmetry. I’m not sure exactly how to do this.
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


