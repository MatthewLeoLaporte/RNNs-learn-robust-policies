---
created: 2024-11-08T11:02
updated: 2024-11-08T11:36
---

## Training networks with contextual inputs

- [x] Some loss plots

### Training methods

#### Direct amplitude information (DAI)

The field strength for each training trial is sampled i.i.d. from a zero-mean normal distribution. 

The network receives the absolute value of the standard normal sample, prior to its scaling by `field_std`.

#### Probabilistic amplitude information (PAI)

The field strength is sampled i.i.d. from a zero-mean normal distribution, and then scaled by `field_std` but also by a uniform sample i.i.d. in $[0, 1]$.

The network receives the value of the uniform sample. Thus it has information about how “on" the field is, i.e. the probability that it will experience a field with std X, versus no field. It does not receive information about the exact strength of the field, on a given trial.

## Plant perturbations

Show that changing the context input controls the robustness of the behaviour.
### Aligned trajectories, varying the context input

#### Trained and evaluated on curl fields, no delay

Evaluation curl fields have amplitude 4.

##### Trained with DAI

- At low field std (0.4), even very high values of the context input do not induce the network to be robust to the amp 4 curl field. Perhaps this makes sense if we consider that the network never 
- Increasing the context input increases the magnitude of the control forces, as expected
- The control force trajectories at intermediate std look pretty similar to [[results-1#^compare-curl-train-aligned|those]] for the part 1 networks
- The context input 0 condition does not look identical to the baseline network condition from part 1; it looks smoother, probably because the network is more robust overall, even when it is not given to expect a disturbance, whereas the baseline network was not exposed to perturbations during training at all
- With increasing context input, the behaviour seems to asymptotically converge on a “most robust” achievable trajectory, which becomes more robust with increasing field std. 
- With increasing std, *everything* becomes relatively more robust – so while the negative context inputs remain “less robust” than the zero context input for that training condition, they may be more robust than the zero context input for a weaker training condition (for example)
- The relationship starts to break down around std 2.0

Std 0.4
![[file-20241126113220236.png]]


Std 1.2
![[file-20241126113300126.png]]

Std 1.6
![[file-20241126113313865.png]]

Std 2.0
![[file-20241126114358068.png]]

###### Comparison of -2, 0, and 2 context input trajectories, across train stds

![[file-20241126121811342.png]]
![[file-20241126121822102.png]]
![[file-20241126121830822.png]]
##### Trained with PAI

- At low std (0.4), high context input results in more robust trajectories than with DAI; this makes sense if we consider that the network is uncertain about how strong exactly will be the field on a given trial
- Negative context inputs are definitely “unrobust” even when we train on std 1.6
- The position trajectories for zero-context look reasonably similar to [[results-1#^compare-curl-train-aligned|those]] from part 1
- However, the force trajectories look somewhat different; in particular there is more context-dependent curvature to the initial forces
- **At std 0.8 and above, values of the context input greater than 1 can induce a “hyperrobust” trajectory, which may be nearly straight on average, or which may even curve in the opposite direction to the curl field.** (Remember, at trial start the network has no idea what direction the curl will be.)
- Very high context inputs lead to a sort of “hyperrobust instability” where the control forces begin oscillating, almost as they do for the “unrobust instability” (i.e. the dark blue curves) but *in the opposite direction*

Std 0.4

![[file-20241126114819811.png]]


Std 0.8
![[file-20241126114917351.png]]

Std 1.6
![[file-20241126115000125.png]]

Std 2.0
![[file-20241126115410900.png]]

###### Comparison of -2, 0, and 2 context input trajectories, across train stds

- Compared to DAI, it is nice that the zero-context trajectories are so similar here; though at higher train stds, they do become smoother
- I’m not sure how to characterize what is happening with the negative context inputs
- It does seem that something qualitative happens between 0.8 and 1.6; the trend at context -2 reverses direction (it stops becoming more unrobust, and starts becoming more robust again); the force profile for context 0 changes significantly; the “hyperrobust oscillations” start becoming apparent in the context 2 position trajectories (though the shift is already apparent in comparison between 0.8 and 1.2 force profiles)


![[file-20241126121955803.png]]
![[file-20241126122005631.png]]
![[file-20241126122026428.png]]

#### Trained and evaluated on curl fields, 4-step delay

**Evaluation curl fields have amplitude 2. Note that this is only half as strong as before, since curl amplitude 4 is very unstable for networks trained on delay 4.**

##### Trained with DAI

- As in the non-delayed case, there is a kind of asymptotic effect with the context input, and the achievable robustness depends on the train std
- Very high values of the context input can smooth out the oscillations, but do not seem to be able to decrease the lateral deviations beyound some bound, even as the train std is increased to the point that things because absolutely unstable

Std 0.4
![[file-20241126135507506.png]]


Std 0.8
![[file-20241126135524479.png]]

The highest context input here is particular interesting
![[file-20241126135809600.png]]

Std 1.2
![[file-20241126135617359.png]]

###### Comparison of -2, 0, and 1 context input trajectories, across train stds

Something weird happens before train std 2.

![[file-20241126143954222.png]]

![[file-20241126144011282.png]]
![[file-20241126144022068.png]]
##### Trained with PAI

- Note that the context-1 case seems to be pretty robust here, but that as we go to values above 1, we appear to more quickly reach “hyperrobust instability” than we did without delay

Std 0.4
![[file-20241126143525172.png]]

Std 0.8
![[file-20241126143559348.png]]

Std 1.2
![[file-20241126143626141.png]]
The context-1 case is very interesting, here:
![[file-20241126143901533.png]]
###### Comparison of -2, 0, and 1 context input trajectories, across train stds
![[file-20241126144135029.png]]
![[file-20241126144146327.png]]

![[file-20241126144157899.png]]
Here it seems clearer that training on perturbations induces “hyper-robustness” more quickly in the presence of delays; these are only the context-1 responses and they are already curving in the opposite direction starting just above std 1.2
### Distributions of performance measures

#### Trained and evaluated on curl fields, no delay

#### Trained and evaluated on curl fields, 4-step delay

## Feedback perturbations

Likewise, show that changing the context input appears to change the feedback gains.

### Aligned trajectories

### Distributions of performance measures

## Dynamical structure

- [ ] **Goal steady-state fixed points**: do they systematically change?
- [ ] **Eigendecomposition of steady-state Jacobians**: do the negative real parts become more negative, as the context input increases?

## 