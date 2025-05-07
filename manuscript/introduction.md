---
created: 2024-11-08T10:51
updated: 2024-11-08T11:07
---



To move effectively in environments whose dynamics are predictable, humans can acquire models of novel dynamics (refs on *de novo* learning), or adjust the parameters of their prior models (refs on adaptation). 
For example, healthy individuals learn to efficiently perform repeated trials of straight, point-to-point reaches, and after the introduction of a consistent leftward force field which persists over subsequent trials, they progressively compensate and return to efficient, straight profiles. 

However, model-based learning and adaptation are not possible when environmental dynamics are inherently unpredictable. 
Perhaps I do not know whether my reach will be perturbed leftward or rightward, or what the magnitude of the perturbation (if any) will be. Then, attempting to compensate for a leftward perturbation of a given magnitude will only help in the particular case where such a perturbation does occur; otherwise, my performance may worsen.

*Robust control* provides a formal account of how, faced with unpredictability in the dynamics of a given task, I can *robusten* my performance *on that task* by making a model-free adjustment to my prior policy. 
A characteristic result of this adjustment is an increase in *reactivity*, or sensory feedback gains: Since the environment is unpredictable, I should have less confidence in my prior (internal) models of dynamics, and place higher weight on the most recent available information.
While the effector is moving, as in straight reaching, a more robust policy also involves an increase in *vigour*, i.e. effector momentum or peak velocities/forces.
These increases in vigour and reactivity effectively reduce the  deviations from the optimal trajectories of the unperturbed task. 
But there is a tradeoff: robust policies are more costly. We should expect agents that minimize cost to prefer less-robust policies, as long as the environment is predictable. 

> [!Note]
> - An increase in limb impedance (e.g. co-contraction) is also observed, though co-contraction alone has less of an effect on robustening than was once supposed. 
> - In humans and other animals, the increasing feedback gains should be observed across the motor hierarchy. For example, in unpredictable environments we should expect both spinal reflex circuits and central state estimators to be tuned towards reactivity.
> 

This tradeoff between efficient and robust policies has been observed in human reaching movements. 
When a sudden perturbation appears on a random trial in a longer sequence of mostly unperturbed trials, individuals’ reaches become more vigorous (and reactive?) on the subsequent, unperturbed trial, and this effect washes out as the consecutive, unperturbed trials continue.

However, while the robustness spectrum has been observed in behaviour, and robust control has been formalized in theory, the neural basis of robustening is unclear. 
Here, we first demonstrate that small, single-layer artificial recurrent neural networks (RNNs) can be induced to implement more-robust reaching movements by being trained on more-uncertain environmental dynamics.
Likewise, 





## Figures
### Model schematic

![[ofc-rnn-schematic_pointmass_context.svg]]

### Adaptation 

![[adaptation-versus-robustness.svg]]
