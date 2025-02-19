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

In each case, the field amplitude may also be scaled by a constant factor (i.e. the `train_std`) on each run.

###### Amplitude scaling factor (PAI-ASF)

The field strength is sampled i.i.d. from a zero-mean normal distribution, and then scaled by a uniform sample i.i.d. in $[0, 1]$. 

The network receives the value of the uniform sample. Because of the product with a standard normal sample, the uniform sample is the standard deviation of the zero-mean normal distribution from which the trial’s field amplitude is sampling. Thus the network has information about the standard deviation, and thus the probability that it will experience a field at least as absolutely strong as any given threshold. It does not receive information about the exact strength of the field, on a given trial.

###### Noisy context (PAI-N)

Similar to [[#Direct amplitude information (DAI)|DAI]], except that Gaussian noise is added to the absolute field amplitude before providing it to the network. Thuis

### Replicates
#### Exclusion from further analysis based on performance measures

The logic here is that systems like the brain will have much more efficient learning systems, and that we are approximating their efficiency by taking advantage of variance between model initializations.

In other words: we are interested in the kind of performance that is feasible with these kinds of networked controllers, more than the kind of performance that we should expect on average (or in the worst case) given the technical details of network initialization etc.

**For this reason, it may be best just to consider the best-performing replicate in each case, except for supplementary analysis where we should the variation between replicates.**

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
### Single-unit stimulation

Outline based on a conversation with Gunnar:

- Perturb a single unit in all the different contexts (e.g. force field strength), resulting in a bunch of different responses for different contexts/context combinations for a single unit
- Observe qualitatively what changes between context. For example, if context only changes the amplitude of the stimulation response but not the direction, then we will boil each response (set of states) to a single number. So we will have one number (e.g. relative amplitude) for each context/context combination; i.e. N numbers for N contexts.
- Do e.g. linear regression to turn N numbers to M numbers, where M is the number of context variables (i.e. get trends for each context variable)
- Repeat this for all the other units
- Now we can e.g. do a scatter plot of the regression parameters across all the units

## Summary of conditions

- 3 train and 3 test perturbation conditions (control, curl, random) such that for each noise+delay condition we can do 3x3 train-test comparisons
- 3 delay conditions (0, 2, 4 steps); these do not vary between 
- 3 noise conditions (0, 0.04, 0.1)