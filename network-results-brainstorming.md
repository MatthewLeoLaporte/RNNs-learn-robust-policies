---
created: 2024-11-07T11:45
updated: 2024-11-11T13:53
---
## Eigenspectra

Part 1 and part 2; easy enough. 

Use the Jacobians around fixed points, since the weight matrix will not reflect the influence of the context input.

## Unit perturbations and tuning

- Change in unit gains with value of context input
### Feedback gains

How do the individual units change their responses to a feedback impulse when the context input changes?

What happens if we send sensory feedback to only a subset of network units?

- Do more-robust networks have stronger weights between these units, and the rest of the network? 
- Does increasing the context input, increase the effective gains between these units and the rest of the network? 

If we perturb these units individually, and we also ensure that there is no direct readout from them, then any effect on the output must be through other units in the network.

### Directional tuning 

Perturb a single unit while reaching in N different directions → compare with the unperturbed-unit reaches → recover the unit’s tuning with respect to movement direction

### Robustness

How robust are hidden trajectories to unit perturbations? 

## Fixed points

- Interpolating the goals-goals and inits-goals manifolds across context inputs
- Do fixed points change location in state space?
- Do attractors steepen with increasing context input?

## Alignment of unit activities and output weights

As in [@SchuesslerEtAl_2024], assuming the network activities $x_{i}(t)$ are mean-subtracted (centered) over time:

$$
\rho= \frac{\lVert\mathbf{W}_{\mathrm{out}}^{\top}\mathbf{X}\rVert}{\lVert \mathbf{W}_{\mathrm{out}} \rVert~\lVert \mathbf{X} \rVert }=\frac{\lVert\mathbf{Z}\rVert}{\lVert \mathbf{W}_{\mathrm{out}} \rVert~\lVert \mathbf{X} \rVert }
$$

where the norm is the Frobenius norm. 

This correlation is large for networks where the top PCs are more strongly correlated with the output weights. Thus this measures how dominated the network is by activity which is orthogonal to the readout.

Then the norm of the outputs can be expressed as a function of the correlation and the norms of the weights and activities:

$$
\lVert \mathbf{Z} \rVert =\rho \lVert \mathbf{W}_{\mathrm{out}} \rVert \lVert \mathbf{X} \rVert 
$$

We can also express this relationship locally in time, assuming one-dimensional outputs and (for the correlation to be valid) that both $\mathbf{w}_{\mathrm{out}}$ and $\mathbf{x}(t)$ are centered *in coordinates*:

$$
\rho(t)=\frac{\mathbf{w}_{\mathrm{out}}^{\top}\mathbf{x}(t)}{\lVert \mathbf{w}_{\mathrm{out}} \rVert~\lVert \mathbf{x}(t) \rVert  }
$$

such that the network’s outputs can be expressed as a function of the local correlation:

$$
\mathbf{z}(t)=\rho(t)\lVert \mathbf{w}_{\mathrm{out}} \rVert ~\lVert \mathbf{x}(t) \rVert 
$$

Because $\lVert \mathbf{x}(t) \rVert$ cannot be small for an RNN, then by controlling the size of $\lVert \mathbf{w}_{\mathrm{out}}  \rVert$, the magnitude of the correlation $\rho$ must compensate to ensure appropriately-sized outputs. Thus the choice of readout weights influences whether the network’s dynamics are dominated by output-orthogonal activity or not. 

### Methods

#### Part 1

- [ ] For networks with trained readout weights, compute the correlation for different networks; see if the correlation/norm of the readout weights depends on the std of disturbances during training
- [ ] Try fixing the readout weights to different norms instead of training them, and see how this affects the robustness measures

#### Part 2

- [ ] Try fixing the readout weights to different norms instead of training them, and see how this affects the network dynamics

## Effective dimensionality

e.g. as measured by number of PCs needed to reconstruct X% of the output.

- Does training on perturbations (vs. not) affect the dimensionality?
- Does the context input?

