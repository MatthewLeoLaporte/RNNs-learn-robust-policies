

- If we focus on a single type of perturbation outside the supplementary analysis, it should probably be SIRF rather than CF. 

## Part 1

### Response to mechanical perturbations

2x2 grid of center-out sets. 

#### Example center-out sets +/- SU

As on the NCM poster, plus PA zero. 

Small task variant.

#### Aligned center-out sets, by SU

Full task variant.

As on the NCM poster, plus PA zero.

#### Velocity profiles, by SU

As on the NCM poster, plus PA zero.

#### Quantification

Full task variant. 

Include the ones on the NCM poster, but maybe also:

- Tangling

### Supplementary

- (Maybe) CF versions of all of the above, if we leave them out of the main results. 
- Comparison of CF and SIRF robustness
	- i.e. what happens if we evaluate CF-trained on SIRF, or vice versa?
- Effect of noise on performance of trained models
- Effect of training on noise
- Frequency response 

## Part 2

### Response to mechanical perturbations

Same as in part 1, except comparisons are by SISU instead of by SU.

As on the NCM poster, plus PA zero, and velocity profiles. 

### Response to feedback perturbations

> At steady state (i.e. already at goal) apply a step-change to the velocity feedback, in a random direction, for 5 time steps. Align the response to impulse direction.

#### Velocity profiles

As on the NCM poster. 

#### Quantification

Max net control force, as on NCM poster. 

Also:

- Tangling, around the time of perturbation. 

#### Supplementary

- Input sensitivity: largest singular value of the Jacobian wrt inputs

### Fixed point analysis

#### PCs vs. SISU

As on NCM poster. **Was this with or without a mechanical perturbation?** We should probably show both. 

**Also**: along with the 3D plots as on the poster, could show 2D plot(s) along the SISU axis (i.e. showing 4 cardinal reaches), which could help to give an intuition for tangling?

#### Stability/contraction analysis

- Jacobian eigendecomposition at steady-state fixed points 
- Input sensitivity along state trajectory (not unsteady fixed points, I think)
- Hessian norm stuff (see TODO & ask o3 about it) - indicates the size of the neighbourhood in which the linearization is valid. Is this relevant?

### Unit stimulation



## Supplementary: Adaptation versus robustness

There is already a bit of evidence for robustness (versus adaptiveness) in that the part 2 networks benefit from being given the SISU. If they were adaptive they’d just estimate the field parameter in the first 5 timesteps or so (no delay!) and then cancel it out more effectively, regardless of the SISU.

### Direction-reversal probe

Start evaluating on a field in one direction and then flip it midway through the trial. The resulting error should be larger if we’re adaptive than if we’re robust. 

Be careful. How can we set this up so that we know that the perturbation will result in a certain magnitude of deviation, depending on robust vs. adaptive?

### Regress late control forces against early perturbation parameter values 

If we’re simply robust, then control forces will be correlated with the SISU but not with the actual value of the field parameter. 

If we’re adaptive, we’ll change our control forces based on our estimate of the actual field parameter.