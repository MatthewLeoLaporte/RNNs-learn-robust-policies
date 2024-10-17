---
created: 2024-10-04T11:27
updated: 2024-10-16T11:05
---
- [x] Move this section into TODO within that repo

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


