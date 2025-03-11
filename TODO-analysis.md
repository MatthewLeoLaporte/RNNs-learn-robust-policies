
- [ ] Try training a significantly larger network in part 2 and see if the context 0 curves overlap more

### Behaviour 

- [ ] See how much more robust the baseline model is in part 1, if we **decrease the weight on the control cost**. (Will also affect the time urgency.)
	- My expectation is that it will of course increase the forces and decrease the movement duration, but that depending on the perturbation this will actually make it *less* robust (e.g. curl fields)
- [ ] In part 2, what is the relationship between train std. and robustness, at a given context input? 

#### Measures

- [ ] **Direction of max net force**
	- at start of trial?

#### Training evolution

- [ ] Animate the aligned trajectories (comparing train std) over training batches

### Network analysis: Unit level

> [!NOTE]
> Using part 2 hybrid models, since otherwise there is no basis from comparison between more- and less-robust networks.

#### Influence of context input on network dynamics 

Module: `part2.context_pert`.
##### Steady-state

- [x] Run a steady-state trial and change the context input (but nothing else) midway through
	- or, ramp the context input across the trial
	- how does the activity change?
	- does the point mass move? hopefully not, but the network was never trained with within-trial variation of the context

##### Reaching

Repeat `context_inputs` for reaching trials that are perturbed with a curl field; compare baseline optimal vs. robust to context-perturbed optimal vs. robust

##### Interpretation

When we perturb context input at steady-state, we observe that the unit activities change, but the point mass more or less does not move. 
This suggests that the direct influence of the context input is in the null space of the network.

When we perturb during a reach, the trajectories are altered and the point mass moves differently?

In either case, when we project into PC space (based on workspace-covering reach trajectories) the context perturbation should *cause* us to move in a null direction.

#### Preferred versus effective directions of units at steady state 

Module: `part2.unit_perts`.

> [!warning] 
> Make sure where the intervention is taking place in Feedbax! Do we perturb before or after the recurrent update? If after, then the immediate consequence of stimulation is entirely determined by the readout, which isn’t very interesting. But we’ll still get the recurrent effects on the next step, of course.

> [!NOTE] 
> Also we can analyze these two individually at steady state, but the preferred directions on their own don’t tell us much.)
> 
> - [ ] **Do tuning curves get narrower** (i.e. do units become less active more quickly as they move away from their preferred direction?) with context input?

Here, our analysis is based on evaluation on two tasks.

1. At the origin steady state, perturb the point mass with a constant force for several time steps. Infer unit preferred directions from their activities at peak force output for a trial.
2. Also at the origin steady state, perturb a network unit with a constant input for several time steps. Repeat for each unit in the network. Infer effective directions as 

##### Preferred directions analysis

- At steady state, perturb the point mass in center-out directions
- At the time step of max forward force (max accel.), find the activities of all units
- Find instantaneous preferences ~~(direction of max activity for each unit; circular distributions of activity for each unit)~~

> [!NOTE] 
> Preferences should not be just argmax, especially since this limits our resolution by the number of directions we perturb in. 
> 
> Instead, use regression to infer a vector in the direction of max activity

##### Effective directions single-unit stimulation analysis

- At steady state, perturb each unit in the network
- Compare the direction of max acceleration, to the preferred direction of the same unit from 1. Note that the instantaneous preferences may not capture the unit-specific effects on recurrent processing, whereas unit perturbations should induce them.
- Do the preferred & perturbed directions align more, for low vs. high context inputs? 
- Also can do other analyses on the perturbed responses...

#### Variation in effective direction, over a reach 

Repeat the effective direction single-unit stimulation analysis, separately at several time points along a reach. 
How do the (distributions of) effective directions of individual units shift over the reach? 
Does this change with context?

- [ ] Also: Variation in preferred versus effective direction, over a reach
- [ ] 
#### Individual unit ablation 

Fix the activity of each unit to zero, in turn.

- [ ] Is performance more sensitive to the ablation of some units, than others? 
- [ ] How does this depend on reach direction? etc?

### Network analysis: Population level

- [ ] Also: repeat all of this for a leaky vanilla RNN

#### Troubleshooting FP finding 

- [ ] **Examine the loss curve over iterations of the fixed point optimization**
	- Get an idea how quickly it converges, if this varies much between conditions, and if there is something to be understood about the FPs that (appear to) fail to converge
	- possibly, plot the hidden state over the optimization, like I did once before (when I showed that the multiple FPs found by the algorithm were actually corresponding to a single FP, by allowing them to converge more)
- [ ] Try to get rid of dislocations in fixed point trajectories (though they aren’t very bad except at context -2)

#### Translation (in)variance 

- [x] Do fixed points vary with goal position (i.e. target input), or just with the difference between target and feedback inputs?
	- for example, do the fixed points change if we translate the target and feedback position inputs in the same way?
	- The “ring” of steady-state (“goal-goal”) FPs for simple reaching suggests this might be the case. 
	- **They do vary.** The grid of steady-state FPS form a planar grid in the space of the top three PCs. As context input changes, this grid translates along an ~orthogonal direction. 
	- **Should plot the readout vector** and see if the direction of translation is roughly aligned with it, which would make sense. 

#### Steady-state FPs 

##### Variation with feedback inputs

i.e. find steady-state FPs, then change the feedback input and see if the location/properties of the FP changes

- [ ] As we increase the perturbation size, do the eigenvectors become aligned? Do they point in the direction in state space that would increase the readout in the corrective direction?
- [ ] **Identify the most unstable eigenvectors. Which units have the strongest contribution?** If we perturb these units at steady state, versus some other randomly selected set of units, do we get a large “corrective” change in the network output?

##### Variation with context input

- [ ] **How do the init-goal and goal-goal fixed point structures vary with context input?**

#### Steady-state Jacobian eigenspectra 

- [ ] What are the “wing” eigenvalues, e.g. seen for the origin steady-state FP, in DAI+curl?
	- Note that they seem to be the strongest oscillatory modes
	- They become larger (i.e. decay more slowly) and higher-frequency with increasing context input 
	- They become relatively larger and higher-frequency when training on stronger field std.

##### Stimulation of Jacobian eigenvectors

- [ ] At the origin steady state FP, perturb the eigenvectors of the Jacobian

When we stimulate them what is different from when we stimulate one of the eigenvectors whose eigenvalue is in the circle around 0? 
How does this change based on context input? e.g. the “circle around origin” doesn’t really exist for the less-robust networks.

#### Steady-state Hessian 

- Are the fixed points minima of the dynamics? 
- Look at the Hessian eigenspectra
- Is there a difference between steady and unsteady FPs? e.g. maybe steady FPs appear as minima, but unsteady ones less so?

#### Variation in effective direction, with directional properties of fixed points 

Repeat the [[#Effective directions single-unit stimulation analysis]] at FPs which have a direction associated with them.
Now, each of these FPs is associated with a non-zero force in some direction. 
How does that direction relate the to the effective direction? 
Do effective directions bend towards the FP force direction? 
Does this happen more or less, in more robust networks?


> [!NOTE]
> This is in this section on population (versus unit) analysis because while it is a single-unit analysis, it is investigating population properties (i.e. of the FP) as well. 
> 
> Thus we can present the single-unit results, then the FPs, then the results that depend on both.

##### Reaches

In the middle of reach trajectories are “unsteady” fixed points which correspond to non-zero force outputs, and which the network usually only approaches and does not pass through.

Here, it might only make sense to do the unit perturbation analysis for a single time step (assuming we perturb before the recurrent update).

##### Static loads

When the point mass has to remain stationary under a static force, the network will be at steady state at a fixed point corresponding to an equal and opposite force. 

Thus, place the point mass at the origin, make its goal the origin, and apply a load it has to statically counter. When it reaches steady state again, perform the effective direction.

#### TMS/tDCS analogues 

Stimulate all the units in the network simultaneously, to mimic TMS/tDCS. Specifically:

- TMS is analogous to adding to the hidden vector (i.e. directly increasing the firing rate)
- tDCS is analogous to adding a small bias term (i.e. in Feedbax, add to the hidden state *after* doing the recurrent update, but before the nonlinearity)

How does this vary with context input? If there are clear differences, then we may be able to predict what will happen when we apply TMS/tDCS.

#### Other 
##### Poincare graph - phase portraits

![[file-20250116174358466.png]]

- [x] see work scratch
	- we graph $\det{A}$ versus $\mathrm{Tr}~A$
	- (but this is for continuous systems I think)
- http://phaseportrait.blogspot.com/2013/01/discrete-time-phase-portraits.html
- for LTI systems, it may make sense to do the discrete → continuous conversion
- https://en.wikipedia.org/wiki/Logarithm_of_a_matrix#Calculating_the_logarithm_of_a_diagonalizable_matrix

If this does work, then:

- [ ] See how context input (and train std? reach directions?) shifts the plot