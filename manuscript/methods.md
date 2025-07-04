---
created: 2024-11-08T10:03
updated: 2024-11-12T10:42
---

## Overview

> [!todo]
> Very general overview paragraph to orient the methods
> 
> “In this project, we were investigating the representation of robustness… to investigate this, we did X… in the next sections we will provide the details”
> 
> In intro we go from general → specific thing we want to achieve; in methods we go from specific thing → details/logic 

Perturbations were fixed for each trial, after sampling their parameter(s) from a given distribution. The distribution of perturbation parameters varied between different ensembles of trained models. 

For example, when the perturbation parameter controls the direction and magnitude of a curl field, we might train one ensemble of models on 

### Part 1: Fixed strategy

**Train one model per perturbation parameter distribution, and provide no explicit information about the perturbation to the network.**

The curl field parameter was sampled i.i.d. from a zero-mean Gaussian distribution. 

### Part 2: Variable strategy

**Train one model per perturbation parameter distribution, but scale the perturbation on each trial by an additional uniformly-sampled (in [0, 1]) signal, which is provided as input to the network.**

The curl field paramer is sampled i.i.d. from a zero-mean normal distribution, and then scaled by an additional uniform sample i.i.d. in $[0, 1]$. The network only receives the value of the uniform sample; it receives no information about the direction or the relative amplitude of the curl field, on a given trial.

The uniform sample represents dynamical uncertainty. If its value is 0, then regardless of the sampled curl parameter, the network’s prior model of the dynamics will be accurate. When its value is 1, then the network should learn to expect a perturbation from the given Gaussian distribution.

Through the product of the uniform sample with the standard normal sample, the uniform sample controls the effective standard deviation of the Gaussian distribution from which the curl parameter is sampled. Thus the network has information about the distribution from which the curl parameter is sampled, or the probability that a curl field of a given amplitude will be present. As this is a signal of how uncertain the network should be in its prior model of the dynamics, we refer to it as *scalar information about uncertainty in the environment* (SIUE).

Or **scalar information about system uncertainty** (SISU).

## Software

> [!notn1]
> Maybe put this in overview
> 

All models were implemented and trained, and all analyses were performed, in Python 3.11. 

JAX + Equinox + Feedbax.

## Models

### System dynamics

In general, the system dynamics under control are described in continuous time by a function $\mathbf{g}$:
$$\.{\mathbf{x}}(t)=\mathbf{g}(\mathbf{x}(t), \mathbf{u}(t))$$

where $\mathbf{x}$ is the vector of states, and $\mathbf{u}$ of control inputs. 

In Feedbax, dynamics are specified in continuous-time form, with discretization performed by the Diffrax library. Thus state evolution equations for the system dynamics will be provided in continuous-time, while iterative equations will be provided for all model components whose solution was not numerically integrated (e.g. forward passes of neural networks).

Due to discretization, we effectively have a state update of the form:
$$\mathbf{x}_{k+1}=\mathbf{g}(\mathbf{x}_{k},\mathbf{u}_{k})$$
where $k$ is the time index such that $k\Delta t=t$.
### Biomechanics

The unperturbed system dynamics are just the dynamics of the biomechanics under control. Here, the dynamics were modeled simply as a Newtonian point mass:
 $$\mathbf{x}=[~\mathbf{p}~~\mathbf{v}~]^{\top}$$
$$\.{\mathbf{x}}(t)\equiv\begin{bmatrix}\.{\mathbf{p}}(t) \\ ~\.{\mathbf{v}}(t)~\end{bmatrix} = \begin{bmatrix}\mathbf{v}(t) \\ ~\mathbf{u}(t)~/m~\end{bmatrix}$$
where $\mathbf{p}$ is the position, $\mathbf{v}$ is the velocity, and $m$ is the mass of the point effector. 

Movements were performed in a 2D workspace, and a drag force $\mathbf{f}_{d}=-k_{d}\mathbf{v}$ was added to model the viscous decay of effector velocity. Thus in Cartesian coordinates, the unperturbed dynamics had the exact linear form:
$$\.{\mathbf{x}}(t)=\mathbf{A}\mathbf{x}(t)+\mathbf{B}\mathbf{u}(t)=\begin{bmatrix}~0&0&1&0~\\~0&0&0&1~\\~0&0&-\frac{k_{d}}{m}&0~\\~0&0&0&-\frac{k_{d}}{m}~\end{bmatrix}\begin{bmatrix}p_x(t)\\p_y(t)\\~v_x(t)~\\v_y(t)\end{bmatrix}+\begin{bmatrix}~0&0~\\~0&0~\\~1/m&0~\\~0&1/m~\end{bmatrix}\begin{bmatrix}f_{\mathrm{net},x}(t)\\~f_{\mathrm{net},y}(t)~\end{bmatrix}$$

Where $\mathbf{f}_{\mathrm{net}}=\mathbf{u}+\mathbf{f}_{\mathrm{pert}}$ . The net force is subject to a zero-order-hold:
$$
\mathbf{f}_{\mathrm{net}}(t)=\mathbf{f}_{\mathrm{net},k},\qquad k\Delta t\le t < (k+1)\Delta t 
$$
> [!Note]
> In Feedbax we approximate everything by Euler’s method after finding the net force on the point mass. However, there are [[Point mass without numerical integration|exact solutions]] for iterating the discrete dynamics of a point mass, including a drag term, and even including a curl field.
> 
> The continuous equation above is thus not quite correct for our implementation. Instead, the drag terms should be moved out of $A$ and into a net force term which includes $u$, drag, and any perturbing forces. 
> 
> A brief mention of this could be made in the discussion. 
> 
> **Mention that we are aware of the matrix exponential being more exact, but that we have chosen a small enough deltaT that it doesn’t really amtter.**

> [!Note]
> Don’t worry about adding in the first-order filter at this point. This is a proof of principle and not a biophysically realistic implementation.
> 

### Mechanical perturbation types

Unless otherwise stated: *mechanical perturbations did not vary within trials.* Either a perturbation with given parameters was present, or not. 
(Do not say that they typically vary between trials, because this is not the case for validation/analyses. It is only really the case for training, so we should state so there.)

**State-dependent forces**: curl.
$$
\mathbf{f_\mathrm{curl}}=\begin{bmatrix}
0 & \phi \\
-\phi & 0
\end{bmatrix}\mathbf{v}
$$
where the value of $\phi$ determines the magnitude and direction (clockwise or counterclockwise) of the field. Note that this is an isotropic field: the magnitude of the curl force does not depend on the direction of $\mathbf{v}$, but only its magnitude. 

**State-independent force**: in general, $\mathbf{f}_{\mathrm{curl}}\in\mathbb{R}^{2}$ is any state-independent perturbing force which was held constant during a trial. How these forces were chosen varied during training and analysis, and is described below. 

> [!TODO]
> This is equivalent to angle uniform on [0, 2pi] and length half-normal. I am not sure whether I should use that representation instead of this one; perhaps it is clearer. 
> 

### State feedback

Delayed, noisy information $\mathbf{y}$ about the state (position and velocity) of the point mass was input to the neural network controller.  
$$
\mathbf{y}_{k}=\mathbf{x}_{k-\delta}+\varepsilon
$$
with additive sensory noise, $\varepsilon \sim \mathcal{N}(0,\sigma_{\mathrm{s}}^{2})$, i.i.d. for each sample.

Separate ensembles of models trained with zero delay, or with **5-step (~50 ms) delay**. 

> [!Note]
> Delay analysis is a single extra figure, not part of the main narrative
> 
### Neural network 

100 gated recurrent units [1].

$$
\mathbf{z}_{k}=\sigma(\mathbf{W}_{\mathrm{ih},z}\mathbf{v}_{k}+\mathbf{W}_{\mathrm{hh},z}\mathbf{c}_{k-1}+\mathbf{b}_{z})
$$
$$
\mathbf{r}_{k}=\sigma(\mathbf{W}_{\mathrm{ih},r}\mathbf{v}_{k}+\mathbf{W}_{\mathrm{hh},r}\mathbf{c}_{k-1}+\mathbf{b}_{r})
$$$$
\^{\mathbf{c}}_{k}=\phi(\mathbf{W}_{\mathrm{ih},c}\mathbf{u}_{k}+\mathbf{W}_{\mathrm{hh},c}(\mathbf{r}_{k}\odot\mathbf{c}_{k-1})+\mathbf{b}_{h})
$$$$
\mathbf{c}_{k}=(1-\mathbf{z}_{k})\odot\mathbf{c}_{k-1}+\mathbf{z}_{k}\odot\^{\mathbf{c}}_{k}
$$
Linear readout.
$$
\mathbf{o}_{k}=\mathbf{W}_{\mathrm{ho}}\mathbf{c}_{k}+\mathrm{bias}
$$
The input to the network concatenates information $\mathbf{s}$ about the movement goal, with state feedback:
$$
\mathbf{v}_{k}=\begin{bmatrix}
~\mathbf{s}_{k}~  \\
~\mathbf{y}_{k}~
\end{bmatrix}
$$
Here, the movement goal was constant over each trial, such that $\mathbf{s}_{k}=\mathbf{s}$. In part 2, an additional scalar input was given.

$$
\mathbf{v}_{k}=\begin{bmatrix}
~\mathbf{s}~  \\
~\mathbf{y}_{k}~ \\
~\varsigma~
\end{bmatrix}
$$

The control signal is obtained from the network output as:
$$\mathbf{u}_{k}=(1+\zeta) \cdot\mathbf{o}_{k}+\xi$$
with additive and multiplicative noise terms $\zeta \sim \mathcal{N}(0, \sigma_{\mathrm{m,a}}^{2})$ and $\xi \sim \mathcal{N}(0,\sigma_{\mathrm{m,m}}^{2})$, where $\sigma_{\mathrm{m,m}}=1.8\,\sigma_{\mathrm{m,a}}$. (Beer, Haggard, Wolpert 2004)

> [!todo]
> Double-check the form of the multiplicative noise

Note that this does not model a distinct efferent transmission delay; the full sensorimotor delay in the system is assumed by the sensory channel.

No noise was added directly to the hidden states

## Training

- Which weights are trained
- Weight initialization
- Adam optimizer
- 10,000 batches × 250 trials.

### Replicates

For each training condition, an ensemble of 5 model replicates was trained, with different random weight initializations.

### Task / training set

What does the training set look like?

Workspace

Undelayed point-to-point reaching between random endpoints in the workspace.

#### Perturbations

**Curl fields**: $\phi\sim \mathcal{N}(0,\sigma_{\mathrm{curl}}^{2})$, where $\sigma_{\mathrm{curl}}$ is varied between model ensembles.

**State-independent force**: 
$$
\mathbf{f_{\mathrm{const}}}=\lambda \begin{bmatrix}
~\cos \theta~ \\
~\sin \theta~
\end{bmatrix}
$$
where $\lambda\sim \mathcal{N}(0,\sigma_{\mathrm{const}}^{2})$ and $\theta\sim \mathcal{U}(-\pi,\pi)$.

### Cost function

Quadratic in position errors, final velocities, control forces, and network activities.

$$
J(\mathbf{x},\mathbf{u},\mathbf{c})=\alpha_{\mathrm{p}}\sum_{k=1}^{K}\left( \frac{k}{K}^{} \right)^{6}\mathbf{p}^{\top}\mathbf{p}+\alpha_{\mathrm{v}}v_{K}^{2}+\alpha_{\mathrm{u}}\sum_{k=1}^{K-1}\mathbf{u}^{\top}\mathbf{u}+\alpha_{\mathrm{h}}\sum_{k=1}^{K}\mathbf{c}^{\top}\mathbf{c}
$$
> [!TODO]
> Double-check that this is correct. Should we use $\left||\mathbf{u}\right||^{2}$ instead of $\mathbf{u}^{\top}\mathbf{u}$?
> 

### Optimality 

> [!Note]
> Just make a short comment that these things didn’t matter.

Performance was not significantly altered by the following measures:

- [x] **Introduce perturbations after initial training period without them**
- [x] Learning rate schedule, e.g. [[2024-11-08|Cosine annealing schedule]] 
- [x] Try `optax.adamw` for [[2024-11-08#Weight decay|weight decay regularization]]
	- Doesn’t make much of a difference
### Hyperparameters

Try training at different network sizes etc.

> [!note]
> Don’t worry about this. Maybe *later* will need to provide a figure showing how performance (loss?) degrades below a threshold of network size

## Analyses

> [!note]
> Just mention all the different analyses and give the details. Which algorithms/implementations did we actually use, what were the parameter values, etc.

### Visualize and quantify robustness (Parts 1 and 2)

Center-out task with fixed perturbation parameters

Curl fields: $\phi=$ some constant value(s).

State-independent fields: $\mathbf{f_{\mathrm{const}}}=$ orthogonal (leftward) to reach direction at some constant amplitude. 

#### Robustness measures

max lateral displacement, max parallel velocity, max net control force, max parallel force
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

## Hardware and cost

- Model evaluation (i.e. to generate states used in analyses) generally took on the order of 15 s to 150 s
- Evaluations results were cached to avoid re-evaluation when modifying/debugging analysis code
- Analyses vary significantly in their execution time. Most are fast (< 30 s) but some take longer (e.g. finding fixed points at low tolerances takes several minutes).

### Titan Xp

Training takes about 10 min per ensemble of 10 models; i.e. about 5 h for 30 ensembles.

> [!todo]
> Test how long an ensemble of 5 models takes to train on the Titan.

### MacBook Pro

M4 Pro with 14-Core CPU, 48 GB RAM

> [!todo]
> Test how long a training run takes.

## Statistical tests 

See [this](https://console.anthropic.com/workbench/f2b0a80f-dfef-4134-b5fa-d7e983fedf99) summary.

I need to test whether some sets of data come from the same distribution or not. I would prefer to use non-parametric tests because I don’t think it is appropriate to assume normality. 

1. Basic test of same distribution: either Wilcoxon rank-sum test or Kolmogorov-Smirnov. Apparently KS is more general, whereas Wilcoxon is based on difference in medians, but I do not understand this formally.
2. Test for difference in variances: Levene’s test, probably. 
3. Also quantify effect sizes. This can be as simple as comparing the means/medians or variances, once a significant difference has been demonstrated. In some cases it might make sense to use [Cohen’s d](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d) or similar.

## Units

See [[units|here]]. 

## Other

- [[Tangling]]
- [[Subspace decomposition methods]]