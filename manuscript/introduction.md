---
created: 2024-11-08T10:51
updated: 2024-11-08T11:07
---
`> [!Note]
>  What minimally do we need to cover so that people will be on board with the approach?
> 

## Overview

**Strengthen the gap statement from the abstract in the first paragraph (or two).**

Effective movement in unpredictable environments demands robustness, or the ability to maintain performance despite uncertain perturbations. Faced with unpredictability, agents can make model-free adjustments to *robusten* their existing motor control policies. Robustening does not mean learning specific dynamics, but enhances performance across a distribution of possible perturbations. 

## Justification and background

**Justify the approach: What is currently known, and what are the debates in the field?** 

### Current understanding of motor cortex

- Low-dimensional / population dynamics on manifolds
- Autonomous versus input driven dynamics;

#### Endogenous dynamics and tangling 

 [[Tangling]] [@RussoEtAl2018;@PerichEtAl2024]

- Tangling describes how predictable the future state of a dynamical system is, given its present state
- The more different the future state of the network on different trials, starting from a more similar initial state, the more tangled the dynamics (the rougher the flow field)
- Chaotic systems have higher tangling than stable systems; so tangling can vary endogenously 
- However, it also varies due to driving by unexpected exogenous disturbances
- In networks whose endogenous activity 

#### Subspace decompositions

### Motor control strategies

If possible, build on stuff from previous section.

- (Maybe) robust control theory
- (Maybe) Difference between *de novo* learning, adaptation, and robustening

#### Definition of robustness

**General: Maintaining performance despite unpredictability of environmental dynamics.**

- Note that this implies the unpredictable dynamics are still in some distribution; we can always find a breaking perturbation. 

[@PerichEtAl2024]
> Deterministic dynamics facilitate  robust movement generation, but flexible motor output requires rapid responses to unexpected inputs.

This uses robustness in a different way than we do. In particular, I think it refers to the *robustness to noise* described in [@RussoEtAl2018]. They demonstrated this robustness by training a network to generate 1) a tangled motor trajectory (a figure eight), **via** 2) a *constrained* hidden state trajectory, which was essentially the figure eight trajectory plus a third dimension, parameterized so they could control the state-space distance between the crossed-over states of the figure eight. When the parameter (i.e. distance) was made to be smaller, then the system was less robust to noise – that is, small variations in the state were more likely to lead to an intersection of state trajectories.

On the other hand, when there are unexpected but lower-frequency exogenous inputs, robustness means *increase control gains*, i.e. input sensitivity.

Note that a more robust model in our sense has higher feedback gains (i.e. stronger response to unexpected inputs).
#### Characteristics of robust motor control policies

A key characteristic of robust policies is their increased *reactivity* (i.e. sensory feedback gains) and *vigour* (e.g. peak forward velocity). ~~This emerges from optimization pressure, and is easy to explain:~~ given the unreliability of internal models in unpredictable contexts, an agent should place more confidence on recent sensory information, and try to minimize errors (both goal errors and disturbance effects) more aggressively.

However, there is a tradeoff: robust policies are more costly. We should expect agents that minimize cost to prefer less-robust policies, as long as the environment is predictable. 
This tradeoff between efficient and robust policies has been observed in human reaching movements. 
When a sudden perturbation appears on a random trial in a longer sequence of mostly unperturbed trials, individuals’ reaches become more vigorous (and reactive?) on the subsequent, unperturbed trial, and this effect washes out as the consecutive, unperturbed trials continue.

#### Strategy tradeoffs in human experiments

[@CrevecoeurEtAl2019]

> [!note]
> Fred & Steve used *a version of robust control*. Do not need to be specific when citing the paper.
> 
> In the methods (and/or discussion) mention that we are not implementing a robust control formalism, but allowing the network to optimize the policy based on cost function (which does not have an explicit robustness term like the formalisms do) and design of the training set (with or without perturbations). However, we could still frame the perturbations as modifications of A/B matrices.  
> 

### RNNs as models of control in the brain

- Refer to recent work, e.g. task-driven modeling studies, particularly using RNNs

[@LillicrapScott2013]
## Our approach 

Should follow continuously from the last section.

**“Here, we…”**

However, while the robustness spectrum has been observed in behaviour, and robust control has been formalized in theory, the neural basis of robustening is unclear. 
Here, we first demonstrate that small, single-layer artificial recurrent neural networks (RNNs) can be induced to implement more-robust reaching movements by being trained on more-uncertain environmental dynamics.
Likewise, …


## Figures
### Model schematic

![[ofc-rnn-schematic_pointmass_context.svg]]

### Adaptation 

![[adaptation-versus-robustness.svg]]
