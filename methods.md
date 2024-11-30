---
created: 2024-11-08T10:03
updated: 2024-11-12T10:42
---

## Models

### Networks

### Biomechanics

- [ ] **Velocity damping**

### Feedback

## Task

Summarize the tasks, but perhaps describe them in more detail in [[#Training]] and [[#Analysis]].

- Simple (not delayed) reaching 

### From notebook 1-2a

We will generally evaluate on a $2\times2$ grid of center-out reach sets (i.e. 4 sets total), with 24 reach directions per set. This is to ensure good coverage and a larger set of conditions/trials on which to perform statistics.

In the case of visualization of center-out sets, we'll use a smaller version of the task with only a single set of 7 reaches (using an odd number helps with visualization).

For 4 sets of 24 center-out reaches (i.e. 96 reach conditions), with 10 replicates and 5 evaluations (i.e. 50 trials) per reach condition, and 100 timesteps, evaluating each task variant leads to approximately 1.5 GB of states. If we run out memory, it may be necessary to:

- reduce the number of evaluation reaches (e.g. `n_evals` or `eval_n_directions`);
- wait to evaluate until we have decided on a subset of trials to plot; i.e. define a function to evaluate subsets of data as needed;
- evaluate on CPU (assuming we have more RAM available).

## Training

- Which parameters are trained
- Parameter initialization
- Adam optimizer
- [[2024-11-08|Cosine annealing schedule]] (isn’t critical for convergence)
### Cost function

- Position and velocity errors
- Control forces (not necessarily network output!)
- Network activity
- [ ] Weight decay?
- [ ] Readout norm?

### Part 1: Single strategy networks

### Part 2: Hybrid strategy networks

#### Training methods

##### Binary context switch (BCS)

The network is simply given a Boolean (0-1) input which indicates whether the training perturbation is currently active, though it vary in amplitude and direction.

##### Direct amplitude information (DAI)

The field strength for each training trial is sampled i.i.d. from a zero-mean normal distribution.

The network receives the absolute value of the standard normal sample, prior to its scaling by `field_std`.

##### Probabilistic amplitude information (PAI)

The field strength is sampled i.i.d. from a zero-mean normal distribution, and then scaled by `field_std` but also by a uniform sample i.i.d. in $[0, 1]$.

The network receives the value of the uniform sample. Thus it has information about how “on" the field is, i.e. the probability that it will experience a field with std X, versus no field. It does not receive information about the exact strength of the field, on a given trial.

### Replicates

- Exclusion of replicates from further analysis based on performance

### Optimality 

It may be necessary to do one or more of the following to get optimal models:

- [ ] **Introduce perturbations after initial training period without them**
- [x] Learning rate schedule
- ~~Batch size schedule (increase later in training)~~ This is essentially equivalent to a learning rate schedule.
- [ ] Gradient clipping
- [x] Try `optax.adamw` for [[2024-11-08#Weight decay|weight decay regularization]]
	- Doesn’t make much of a difference

### Hyperparameters

Try training at different network sizes etc.

### Hardware and cost

Titan Xp

Training takes about 10 min per ensemble of 10 models; i.e. about 4 h for 30 models

## Analysis

### Validation task

### Robustness measures

### Feedback perturbations

- How to make pos vs. vel perturbations comparable?
- Choose amplitudes to align the max (or sum?) deviation for the control (zero train std) condition?

## Summary of conditions

- 3 train and 3 test perturbation conditions (control, curl, random) such that for each noise+delay condition we can do 3x3 train-test comparisons
- 3 delay conditions (0, 2, 4 steps); these do not vary between 
- 3 noise conditions (0, 0.04, 0.1)