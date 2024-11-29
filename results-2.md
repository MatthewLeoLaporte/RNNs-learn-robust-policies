---
created: 2024-11-08T11:02
updated: 2024-11-08T11:36
---
## Training networks with contextual inputs

- [x] Some loss plots

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

###### Same comparison but for disturbance amplitude 2 (instead of 4)

This makes it clearer that 

- hyperrobustness does not occur with this training method, and in fact not much happens when we increase the context above 1;
- there is significant variation in robustness for different train stds, at context 0 (compare this to the PAI method below, where stds up to 1.2 are all pretty similar for context 0)

Context -2
![[file-20241128111504513.png]]
Context 0
![[file-20241128111521393.png]]
Context 1
![[file-20241128111543963.png]]
Context 2
![[file-20241128111610673.png]]

##### Trained with PAI


> [!NOTE]
> I’ve replaced the figures below with the ones for amplitude 2, rather than amplitude 4, since the show the trends more clearly. 


- At low std (0.4), high context input results in more robust trajectories than with DAI; this makes sense if we consider that the network is uncertain about how strong exactly will be the field on a given trial
- Negative context inputs are definitely “unrobust” even when we train on std 1.6
- The position trajectories for zero-context look reasonably similar to [[results-1#^compare-curl-train-aligned|those]] from part 1
- However, the force trajectories look somewhat different; in particular there is more context-dependent curvature to the initial forces
- **At std 0.8 and above, values of the context input greater than 1 can induce a “hyperrobust” trajectory, which may be nearly straight on average, or which may even curve in the opposite direction to the curl field.** (Remember, at trial start the network has no idea what direction the curl will be.)
- Very high context inputs lead to a sort of “hyperrobust instability” where the control forces begin oscillating, almost as they do for the “unrobust instability” (i.e. the dark blue curves) but *in the opposite direction*

Std 0.4
![[file-20241128112530988.png]]

Std 0.8
![[file-20241128112540425.png]]

Std 1.2
![[file-20241128112551445.png]]

Std 1.6
![[file-20241128112622769.png]]


###### Comparison of -2, 0, and 2 context input trajectories, across train stds


> [!NOTE]
> These are for amplitude 4, and are harder to interpret than the results for amplitude 2; see next section.


- Compared to DAI, it is nice that the zero-context trajectories are so similar here; though at higher train stds, they do become smoother
- I’m not sure how to characterize what is happening with the negative context inputs
- It does seem that something qualitative happens between 0.8 and 1.6; the trend at context -2 reverses direction (it stops becoming more unrobust, and starts becoming more robust again); the force profile for context 0 changes significantly; the “hyperrobust oscillations” start becoming apparent in the context 2 position trajectories (though the shift is already apparent in comparison between 0.8 and 1.2 force profiles)


![[file-20241126121955803.png]]
![[file-20241126122005631.png]]
![[file-20241126122026428.png]]
###### Same comparison but for curl amplitude 2 (instead of 4)

This looks much clearer.

Context input -2
![[file-20241128110348679.png]]

Context 0. Notice that up to std 1.2, the trajectories are very similar, which is what we expect to see if the network is really internally separating baseline (context 0) versus robust strategies
![[file-20241128110400104.png]]

Context 1: Now the influence of train std. is apparent, since it determines what level of robustness context 1 corresponds to. Note that the control profiles are qualitatively quite different than for context 0.

![[file-20241128110536141.png]]
Context 2: Here the hyperrobustness is apparent, even as early as std 0.8
![[file-20241128110732123.png]]


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

#### Trained and evaluated on random fields, no delay

##### Determining the train stds. to compare

This was less obvious to me than with curl fields. 

- The switch to a robust strategy happens at quite low field std, and saturates at stds not much higher. 
- 


### Distributions of performance measures for PAI

#### Trained and evaluated on curl fields, no delay

![[file-20241128173957248.png]]

![[file-20241128174025534.png]]

> [!NOTE] 
> Since the PAI model shows "hyperrobust" responses that curve in the opposite direction, perhaps we should use max signed response


#### Trained and evaluated on curl fields, 4-step delay

## Feedback perturbations

Likewise, show that changing the context input appears to change the feedback gains.

### Aligned trajectories

### Distributions of performance measures

## Dynamical structure

- [ ] **Goal steady-state fixed points**: do they systematically change?
- [ ] **Eigendecomposition of steady-state Jacobians**: do the negative real parts become more negative, as the context input increases?

## Other/supplementary analyses

### Effect of delay on robustness scaling

As we increase the feedback delay, how does the relationship between X and the context input change?

Be decisive about what X is.

### Output correlations

The readout norm will be fixed for each hybrid network; however, it may be the case that the output correlation (i.e. partitioning of activity between null and potent spaces) will vary with the context input. 

Quantify this.

### Difference between DAI and PAI 

I think for the main analysis I will focus on PAI, since it allows us to much more effectively control the level of robustness by varying the context input.

However, it would be worth discussing (& including a supplementary figure) about how DAI does not induce robustness in the same way as PAI. 

With DAI we get 1) asymptotic effects, not much happens above context 1, and in particular we can’t get straight or hyperrobust performance; 2) behaviour varies significantly at context 0, across train std, whereas with PAI it is more similar (i.e. it is more of a baseline, irrespective of the perturbations experienced outside the context 0 condition)

This is probably because with PAI the network is always uncertain about how strong the fields will actually be, since its information is probabilistic, and therefore it has to hedge more strongly against this uncertainty, such that it extrapolates better re: the robustness tradeoff.

### Floor on control forces? Minimum work needed to complete task?

This is for the zero-noise, curl amplitude 2 condition, trained on curls. Note that as we increase train std and context input, the integral over the absolute lateral forces bottoms out around 20. Is this because there is a minimum amount of work that is necessary to complete the task, under these conditions, which we are approaching by adopting a more robust strategy?

![[file-20241128121329199.png]]