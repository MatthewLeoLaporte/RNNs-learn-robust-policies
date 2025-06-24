---
created: 2024-11-08T10:03
updated: 2024-11-12T10:42
---

## Inducing model uncertainty



### Part 1: Single strategy networks

The curl field parameter was sampled i.i.d. from a zero-mean Gaussian distribution. 

Multiple ensembles of 

### Part 2: Hybrid strategy networks

The curl field paramer is sampled i.i.d. from a zero-mean normal distribution, and then scaled by an additional uniform sample i.i.d. in $[0, 1]$. The network only receives the value of the uniform sample; it receives no information about the direction or the relative amplitude of the curl field, on a given trial.

The uniform sample represents dynamical uncertainty. If its value is 0, then regardless of the sampled curl parameter, the network’s prior model of the dynamics will be accurate. When its value is 1, then the network should learn to expect a perturbation from the given Gaussian distribution.

Through the product of the uniform sample with the standard normal sample, the uniform sample controls the effective standard deviation of the Gaussian distribution from which the curl parameter is sampled. Thus the network has information about the distribution from which the curl parameter is sampled, or the probability that a curl field of a given amplitude will be present. As this is a signal of how uncertain the network should be in its prior model of the dynamics, we refer to it as *scalar information about uncertainty in the environment* (SIUE).

Or **scalar information about system uncertainty** (SISU).

## Model

### System dynamics

We use a discrete-time representation
$$\mathbf{x}_{k+1}=\mathbf{f}(\mathbf{x}_{k},\mathbf{u}_{k},k)$$

The baseline (unperturbed) system dynamics are just the dynamics of the biomechanical model being controlled.

### Biomechanics

Point mass. Discrete:

$$x_{t+1}=\mathbf{A}x+\mathbf{B}u=\begin{bmatrix}~1&0&\Delta t&0~\\~0&1&0&\Delta t~\\~0&0&1&0~\\~0&0&0&1~\end{bmatrix}\begin{bmatrix}p_x\\p_y\\~v_x~\\v_y\end{bmatrix}+\begin{bmatrix}~\Delta t^2/2m & 0~\\~ 0 & \Delta t^2/2m~\\~\Delta t/m & 0~\\~0 & \Delta t/m~\end{bmatrix}\begin{bmatrix}~f_x~\\f_y\end{bmatrix}$$

With drag; e.g. in the continuous case.
$$\mathbf{f}_{d}=-k_{d}\mathbf{v}$$
$$\frac{d}{dt}x=\begin{bmatrix}~0&0&1&0~\\~0&0&0&1~\\~0&0&-\frac{k_{d}}{m}&0~\\~0&0&0&-\frac{k_{d}}{m}~\end{bmatrix}\begin{bmatrix}p_x\\p_y\\~v_x~\\v_y\end{bmatrix}+\begin{bmatrix}~0&0~\\~0&0~\\~1/m&0~\\~0&1/m~\end{bmatrix}\begin{bmatrix}f_x\\~f_y~\end{bmatrix}$$
### Feedback

Position and velocity. Separate models trained with zero delay, or with 2-step (~20 ms) delay. 

### Network

100 gated recurrent units [1].

TODO: Use something other than $\mathbf{x}$ to avoid confusion with system state.


$$
\mathbf{z}_{t}=\sigma(\mathbf{W}_{\mathrm{ih},z}\mathbf{u}_{t}+\mathbf{W}_{\mathrm{hh},z}\mathbf{x}_{t-1}+\mathbf{b}_{z})
$$
$$
\mathbf{r}_{t}=\sigma(\mathbf{W}_{\mathrm{ih},r}\mathbf{u}_{t}+\mathbf{W}_{\mathrm{hh},r}\mathbf{x}_{t-1}+\mathbf{b}_{r})
$$$$
\^{\mathbf{x}}_{t}=\phi(\mathbf{W}_{\mathrm{ih},x}\mathbf{u}_{t}+\mathbf{W}_{\mathrm{hh},x}(\mathbf{r}_{t}\odot\mathbf{x}_{t-1})+\mathbf{b}_{h})
$$$$
\mathbf{x}_{t}=(1-\mathbf{z}_{t})\odot\mathbf{x}_{t-1}+\mathbf{z}_{t}\odot\^{\mathbf{x}}_{t}
$$
Linear readout.

$$
\mathbf{o}_{t}=\mathbf{W}_{\mathrm{ho}}\mathbf{x}_{t}
$$

### System noise

Gaussian. Sensory → additive. 

Motor → additive + multiplicative.

## Units

See [[units|here]]. 

> [!Note]
> I guess this shouldn’t be a separate section, but that we should just mention the values + units for given quantities, wherever they are first introduced.
> 


## Software

Python; JAX + Equinox + Feedbax.

## Training

- Which weights are trained
- Weight initialization
- Adam optimizer
- 10,000 batches × 250 trials.

### Replicates

For each training condition, an ensemble of 5 model replicates was trained, with different random weight initializations.

### Task 

Undelayed point-to-point reaching. 

### Cost function

Quadratic in position errors, final velocities, control forces, and network activities.

### Optimality 

Performance was not significantly altered by the following measures:

- [ ] **Introduce perturbations after initial training period without them**
- [x] Learning rate schedule, e.g. [[2024-11-08|Cosine annealing schedule]] 
- [x] Try `optax.adamw` for [[2024-11-08#Weight decay|weight decay regularization]]
	- Doesn’t make much of a difference
### Hyperparameters

Try training at different network sizes etc.

### Hardware and cost

#### Titan Xp

Training takes about 10 min per ensemble of 10 models; i.e. about 4 h for 30 models

#### MacBook Pro

M4 Pro with 14-Core CPU, 48 GB RAM

## Analyses

### Visualize and quantify robustness (Parts 1 and 2)

Center-out task with consistent perturbations 

Robustness measures: max lateral displacement, max parallel velocity, max net control force, max parallel force
#### Task description

Evaluate on a  grid of center-out reach sets (i.e. 4 sets total), with 24 reach directions per set. This is to ensure good coverage and a larger set of conditions/trials on which to perform statistics.

In the case of visualization of center-out sets, we'll use a smaller version of the task with only a single set of 7 reaches (using an odd number helps with visualization).

For 4 sets of 24 center-out reaches (i.e. 96 reach conditions), with 10 replicates and 5 evaluations (i.e. 50 trials) per reach condition, and 100 timesteps, evaluating each task variant leads to approximately 1.5 GB of states. If we run out memory, it may be necessary to:

- reduce the number of evaluation reaches (e.g. `n_evals` or `eval_n_directions`);
- wait to evaluate until we have decided on a subset of trials to plot; i.e. define a function to evaluate subsets of data as needed;
- evaluate on CPU (assuming we have more RAM available).
#### Part 1

Compare models trained on different training std.

#### Part 2 

Compare different SIUE

### Feedback perturbations (Part 2)

- How to make pos vs. vel perturbations comparable?
- Choose amplitudes to align the max (or sum?) deviation for the control (zero train std) condition?
### Single-unit stimulation (Part 2)

Outline based on a conversation with Gunnar:

- Perturb a single unit in all the different contexts (e.g. force field strength), resulting in a bunch of different responses for different contexts/context combinations for a single unit
- Observe qualitatively what changes between context. For example, if context only changes the amplitude of the stimulation response but not the direction, then we will boil each response (set of states) to a single number. So we will have one number (e.g. relative amplitude) for each context/context combination; i.e. N numbers for N contexts.
- Do e.g. linear regression to turn N numbers to M numbers, where M is the number of context variables (i.e. get trends for each context variable)
- Repeat this for all the other units
- Now we can e.g. do a scatter plot of the regression parameters across all the units

### State-space analysis 

PCA

Eigendecomposition of fixed point Jacobians 

Validation trajectories: How do FPs change with reach condition and SIUE?

Perturbation responses: How do 
## Summary of conditions

- 3 train and 3 test perturbation conditions (control, curl, random) such that for each noise+delay condition we can do 3x3 train-test comparisons
- 3 delay conditions (0, 2, 4 steps); these do not vary between 
- 3 noise conditions (0, 0.04, 0.1)