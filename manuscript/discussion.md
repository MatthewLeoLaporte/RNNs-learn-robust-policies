---
created: 2024-11-08T10:50
updated: 2024-11-08T23:39
---
Our results show that a single-layer RNN can vary the robustness of its reaching policy through systematic variations in its neural properties, driven by information about UE. 

While we have not determined whether or how the brain computes an analog of the SIUE, we do demonstrate that a continuum of policies of varying robustness can be generated through a simple and general computational shift (**be specific**) which could be implemented locally in the brain, and we provide specific, testable predictions about the neural activity in brain areas which may do so. 

More broadly, our results suggest that local populations of neurons across the brain may adjust their robustness or **reactivity** to ~~**external influences**, including the influences of~~ other neural populations, according to information about their unpredictability.
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

## Neurophysiology

### Scaling of unit gains 

#### In response to mechanical perturbations

See [[Neural tuning#Mechanical perturbation at steady state|here]]. Basically, 

1. Manipulate environmental unpredictability as an analogue of the unpredictability signal.
2. Measure baseline force direction preferences in a center-out reaching task, or during center-out perturbation of a postural stabilization task.
3. Measure activity after a fixed perturbation, across unpredictability contexts.

#### In response to feedback perturbations

Do a visual perturbation, e.g. a target jump, and examine the magnitude of neural responses across unpredictability contexts. 

This is related to the [[#Scaling of input-driving|next section]]. Perhaps it belongs there?

### Scaling of input-driving

#### Tangling

If tangling is related to being input-driven, does that mean that we should expect higher tangling in more-unpredictable contexts? Where should we expect this increase in tangling to appear? 

#### Others

Do any of the following change with environmental unpredictability?

- Changes in membrane excitability or other local properties that change responsiveness to synaptic inputs
- Changes in thalamocortical loops, or the activity of interneurons that gate information flow.
- Changes in oscillatory activity, e.g. decrease in beta band power due to a potential association of beta oscillations with maintaining the current motor policy.

This could be partly justified by the feedback gain results. Additional justification might be possible by examining how the reset/update gate activity of the GRU network changes with unpredictability context, however this would not provide a direct analogy to any of the above, I think. 

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

### Separation principle / network partitioning

We use undifferentiated networks. 

However, certain things are harder to investigate in this context.

- What does the network’s forward model look like? For example, in the case of adaptation (CW curl) vs. robustening (mixed direction curl) what is the difference in the effect on the forward model?

Note that in the future we can approach this problem without needing entirely distinct networks. For example, weight partitioning / unit subsets.

Another option is to explicitly separate the network into policy and state estimation layers.

### Biomechanics 

As our biomechanical model is a point mass, there is no distinction between proprioceptive and visual feedback. There is only a distinction in terms of feedback noise and delay. 

Instead of approximating the biomechanics by using Euler’s method and a net force calculation, we could solve for the discrete iterations [[Point mass without numerical integration|exactly]], even (apparently) in the case that we have drag force and a curl field.

### Replicates and learning

- We are mainly concerned with what performance is *possible*
- The variance of performance and policies among replicates indicates 
- Thus when showing single-replicate data, we choose the best replicate for a training condition
- We also exclude replicates which perform much worse than the best replicate

#### Exclusion from further analysis based on performance measures

The logic here is that systems like the brain will have much more efficient learning systems, and that we are approximating their efficiency by taking advantage of variance between model initializations.

In other words: we are interested in the kind of performance that is feasible with these kinds of networked controllers, more than the kind of performance that we should expect on average (or in the worst case) given the technical details of network initialization etc.

**For this reason, it may be best just to consider the best-performing replicate in each case, except for supplementary analysis where we should the variation between replicates.**