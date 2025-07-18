---
created: 2024-11-08T10:51
updated: 2024-11-08T11:07
---
> [!Note]
>  What minimally do we need to cover so that people will be on board with the approach?
> 

## Overview

**Strengthen the gap statement from the abstract in the first paragraph (or two).**

To win their way through uncertain environments, agents should implement robust policies. 
Both primates+ and artificial neural networks (ANNs)+ learn, and can adapt their motor policies to compensate for ongoing perturbations.
Yet not all perturbations are predictable in time for a precise adaptation.
When humans anticipate such a perturbation, we increase the vigour and reactivity of our movements to mitigate its effect, regardless of its precise form. 

Robustening does not mean learning the specifics of unknown dynamics, but enhances performance across a distribution of possible perturbations. 

## Justification and background

**Justify the approach: What is currently known, and what are the debates in the field?** 

### Current understanding of motor cortex

- populations of many neurons coordinate their activity to [@KalaskaEtAl1997;@KakeiEtAl1999;@GeorgopoulosEtAl1982;@CheneyFetz1980;@ShenAlexander1997]
- Facilitates “robust movement generation” [@RussoEtAl2018;@SussilloEtAl2015]
	- also: neuronal death [@Pakkenberg2003], unreliable synapses [@TsodyksMarkram1997]
- when they are undisturbed, they generate dynamics which are coherent, or intrinsically predictable
	- i.e. describing smooth trajectories on manifolds which are relatively low-dimensional [@SadtlerEtAl2014;@GallegoEtAl2018;@GallegoEtAl2017;@ElsayedEtAl2016;@KaufmanEtAl2014] (relative to the number of neurons in the population)
	- “well-described as a feedforward system with deterministic dynamics” [@MichaelsEtAl2016;@ChurchlandEtAl2012]
	- “undisturbed”: means “no unpredicted inputs”, not “no inputs” 
		- dynamics may still appear coherent if inputs are predictable [@SeelyEtAl2016;@SussilloEtAl2015]
			- i.e. if the inputs match what the network should expect given its outputs and its model of the system it is controlling
		- importantly, “motor cortex in mice requires constant input as it generates behaviour” [@SauerbreiEtAl2020]
- however, motor cortex is only one part of the closed loop which generates behaviour [@PruszynskiEtAl2011;@ScottEtAl2015]
	- it cannot be entirely autonomous, but must respond to sensory feedback
		- “its neurons respond rapidly to limb perturbations and other sensory inputs” [@CheneyFetz1984;@PruszynskiEtAl2011;@SauerbreiEtAl2020]
	- unpredictable inputs may drive the network over trajectories it would not follow spontaneously
	- error-correction 
- input-driven dynamics are confined to a distinct subspace [@PerichEtAl2024]
	- allows the dynamics on the intrinsic manifold (different subspace) to proceed, while integrating the effects of the disturbance affecting the input subspace

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

**In general, robustness may refer to the ability of an agent to maintain performance, despite unpredictability of environmental dynamics.** 
Crucially, no agent can tolerate totally unbounded perturbations. It is always possible to find a breaking perturbation, such as a large enough noise injection or physical push.
So, to what manner of unpredictability is an agent robust? What distribution can a perturbation be sampled from, such that not starting with any more specific knowledge of the perturbation, the agent can still complete its task? 

> [!Note]
> Importantly, to quantify robustness we frame this more like “given a fixed distribution of perturbations and two agents performing the same task, which agent’s performance is less degraded by perturbations from that distribution?”
> 

An agent may be robust to unstructured perturbations such as *Gaussian noise*.
In this case, robustness mostly entails safeguarding the existing deterministic structure of the agent’s policy.
For example, when that policy is implemented by a dynamical controller whose internal trajectories are already nearly tangled, then any small, unstructured perturbation may be sufficient to divert the current operating trajectory and cause a discontinuity in behaviour [@RussoEtAl2018]. 
A more-robust version of this controller might spread out its internal trajectories (i.e. reduce tangling) sufficiently to eliminate such discontinuities. 

[@PerichEtAl2024]
> Deterministic dynamics facilitate  robust movement generation, but flexible motor output requires rapid responses to unexpected inputs.

> [!Note]
> [@RussoEtAl2018] demonstrated noise robustness by training a network to generate 1) a tangled motor trajectory (a figure eight), **via** 2) a *constrained* hidden state trajectory, which was essentially the figure eight trajectory plus a third dimension, parameterized so they could control the state-space distance between the crossed-over states of the figure eight. When the parameter (i.e. distance) was made to be smaller, then the system was less robust to noise – that is, small variations in the state were more likely to lead to an intersection of state trajectories.
> 

On the other hand, perturbations’ unpredictability may be structured.
A participant in a reaching experiment may anticipate that the current trial will be disturbed in a structured way, but have no information about the exact parameters. 
~~If they have a model of the distribution the perturbation parameters will be sampled from~~, ~~if their model for the task is close to optimal, and they cannot learn the remainder~~, then robustness entails *increasing their control gains* [@CrevecoeurEtAl2019]. 
Two signatures of this are increased *feedback reactivity* (i.e. sensory gains) and *movement vigour* (e.g. peak forward velocity).

There is a tradeoff in the response to structured versus unstructured unpredictability. 
When an agent knows that its internal models are *structurally* incomplete or inaccurate, and that there will be structured variability in the environment, it should place more confidence on recent sensory information, and respond to errors more aggressively.
But responding more aggressively is counterproductive when the environment’s deviations from nominal predictions are merely unstructured. 

> [!important] 
> See [here](https://console.anthropic.com/workbench/cd413896-4458-49c4-bf25-c7c9f2eb15d4) for a discussion on the distinction between unstructured (noise) and structured perturbations. 

There is also a tradeoff when tuning control gains: higher-gain policies are more costly. 
We should expect agents that are more cost-sensitive to prefer lower-gain as long as the environment is predictable. 
This tradeoff between efficient and robust policies has been observed in human reaching movements. 
When a sudden perturbation appears on a random trial in a longer sequence of mostly unperturbed trials, individuals’ reaches become more vigorous (and reactive?) on the subsequent, unperturbed trial, and this effect washes out as the consecutive, unperturbed trials continue.

#### Strategy tradeoffs in human experiments



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
