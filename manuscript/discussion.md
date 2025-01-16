---
created: 2024-11-08T10:50
updated: 2024-11-08T23:39
---

## Questions 

What are the emergent properties of a system that learns to be robust? 

- That is, what is the effect of model uncertainty during training, on the learned network policies for motor control?

What is the difference between the “general” robustness of an H-infinity controller, and the “more specific” robustness induced by training on a particular kind of disturbance? 

- We expect some things to be similar (e.g. higher max forward force)
- But some things are different; and this is due to there being differences in what is actually modelable about the disturbance

Can we separate the aspect of the disturbance that is modelable from the aspect that is not? Or if that

Is robustness an all-or-nothing phenomena? No. 

- We can tune the robustness of an H-infinity controller by adjusting the bound $\gamma$.
- We can induce more robustness in RNNs by training on uncertain dynamics.

## Results

Demonstrate that in terms of measures, the models trained on perturbations are actually more robust. 

- However, there are some caveats when comparing these results to the changes in robustness measures we expect from (say) an H-infinity controller
- This is because the H-infinity controller 

### Efficiency and control forces

The maximum control forces tend to increase with robustness, but often the sum of overall forces decreases. 

For example, consider these plots of the max and sum net control force, across changing context input, for models trained on curl fields, and evaluated on amplitude 2 curl fields:

![[file-20241128121846572.png]]

Similar effects are seen for non-hybrid networks trained on different curl stds.

The explanation here is that control gains go up for more robust networks, and so we expect their *worst-case local efficiency* to be worse, however overall they may be more efficient in terms of total expenditure because their strategies more effectively solve the task.

## Limitations and concerns

### Readout norm and null space

Cite study that talks about readout norms and output correlations. 

Discuss how partitioning of activity between the null and potent spaces may be a strategy for robustness.

This deserves further attention in future work.

### Interpretation of fixed points

#### And their Jacobians

It makes technical sense to treat a linearization of a *steady state* fixed point as the actual dynamics of the system, when noise is not high.

When the system is in an unsteady closed loop state (e.g. on its way to a target position) but we compute the fixed points as though the system is in a steady state (or, in open loop) then the fixed points we calculate are not necessarily ever reached by the system in practice. The system is merely falling toward those fixed points, from some point in state space. 

### Types of perturbations studied

##### Why not accelerant/retardant fields, or random velocity-dependent fields?

Accelerant/retardant fields are not very interesting; they either stabilize or destabilize the steady-state attractors.

Random velocity-dependent fields are just some interpolation between an accelerant and a curl field.

### Separation principle

We use undifferentiated networks. 

However, certain things are harder to investigate in this context.

- What does the network’s forward model look like? For example, in the case of adaptation (CW curl) vs. robustening (mixed direction curl) what is the difference in the effect on the forward model?

Note that in the future we can approach this problem without needing entirely distinct networks. For example, weight partitioning.

Another option is to explicitly separate the network into policy and state estimation layers.

### Replicates and learning

- We are mainly concerned with what performance is *possible*
- The variance of performance and policies among replicates indicates 
- Thus when showing single-replicate data, we choose the best replicate for a training condition
- We also exclude replicates which perform much worse than the best replicate