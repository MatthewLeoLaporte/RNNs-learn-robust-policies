---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
    path: /Users/mll/Main/10 Projects/10 PhD/41 RNNs learn robust policies/.venv/share/jupyter/kernels/python3
---

# Network unit perturbations

## Brainstorming

### Single-unit tuning

Perturb each unit at a time and see how their "tuning" changes. 

- i.e. at different points in a reach (including steady-state) what is the effect of perturbing the unit? 
- We might expect certain units to prefer certain directions in general (i.e. distribution persists across reach conditions) 
- Some tunings may also change entirely depending on the reach condition, and on the time point during the reach.

#### Plots

##### Change in tuning distribution with reach direction and time

e.g. a violin plot for a sample unit, where x axis is time; ideally one of those plots where the violin is split, sits "up" in a pseudo-3D, and adjacent timesteps overlap (looks like mountains)

#### What variable to perturb 

The input to a unit, versus the output of the unit? 

1. Perturbing the input may tell us the unit's effect on performance with respect to its inputs (i.e. including its own gain)
2. Perturbing the output tells us the effect of the unit's output on performance. This is more straightforward to interpret, I think.

### Dynamical modes at fixed points

We've already solved for eigenspectra of the network, at fixed points. 

The eigenvectors correspond to weighted sets of network units that support the linear dynamical modes of the network, near the fixed points. 

We should perturb the network units along these modes to understand the dynamics better. 

For example, choose 

1) the most stable non-oscillatory mode (positive real line nearest to 0), 
2) the most unstable mode (positive real line furthest from 0), 
3) a strong oscillatory mode (e.g. the further eigenvalue in the "wing" region)

and perturb each of these when the network is at the steady state.

Thus: 

- when we perturb an oscillatory mode, do we get oscillations around the fixed point?
- do different groups of units contribute to the stability of the fixed point at different steady states (e.g. if the fixed points are on opposite sides of the origin, does the "stability vector" look opposite somehow? does it have a principled relationship with the unit tuning?)
