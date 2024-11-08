---
created: 2024-11-08T10:50
updated: 2024-11-08T11:23
---

## Questions 

What are the emergent properties of a system that learns to be robust? 

- That is, what is the effect of model uncertainty during training, on the learned network policies for motor control?

What is the difference between the “general” robustness of an H-infinity controller, and the “more specific” robustness induced by training on a particular kind of disturbance? 

- We expect some things to be similar (e.g. higher max forward force)
- But some things are different; and this is due to there being differences in what is actually modelable about the disturbance

## Results

Demonstrate that in terms of measures, the models trained on perturbations are actually more robust. 

- However, there are some caveats when comparing these results to the changes in robustness measures we expect from (say) an H-infinity controller
- This is because the H-infinity controller 



## Limitations and concerns

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