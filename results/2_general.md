---
created: 2024-11-08T11:02
updated: 2024-11-08T11:36
---
## Training networks with SISU

- [x] Some loss plots

## Plant perturbations

Show that changing the SISU controls the robustness of the behaviour.
### Aligned trajectories, varying the SISU

#### Trained and evaluated on curl fields, no delay

Evaluation curl fields have amplitude 2.

##### Trained with BCS-75

###### Train std 0.0
![[file-20241214105858243.png]]

###### Train std. 0.5

As with PAI-ASF, at SISU -3 we have some unrobust behaviour even in the absence of curl.
![[file-20241214105913680.png]]
![[file-20241214105953597.png]]

###### Train std 1.0

The “hyperrobust” oscillations in control force show up much earlier and more clearly than in PAI-ASF. However, unlike PAI-ASF, there are no strange oscillations in the control for for SISU 2+

![[file-20241214110010210.png]]

###### Train std. 1.5

The hyperrobust behaviour at high SISUs disappears.
![[file-20241214110058954.png]]
###### Comparison of -2, 0, and 2 SISU trajectories, across train stds

Compared to PAI-ASF, the spread between train stds is somewhat more similar comparing SISUs.

![[file-20241214110229948.png]]
![[file-20241214110238678.png]]
![[file-20241214110437259.png]]
##### Trained with DAI


> [!NOTE]+
> This section has disturbance amplitude 4


- At low field std (0.4), even very high values of the SISU do not induce the network to be robust to the amp 4 curl field. Perhaps this makes sense if we consider that the network never 
- Increasing the SISU increases the magnitude of the control forces, as expected
- The control force trajectories at intermediate std look pretty similar to [[results-1#^compare-curl-train-aligned|those]] for the part 1 networks
- The SISU 0 condition does not look identical to the baseline network condition from part 1; it looks smoother, probably because the network is more robust overall, even when it is not given to expect a disturbance, whereas the baseline network was not exposed to perturbations during training at all
- With increasing SISU, the behaviour seems to asymptotically converge on a “most robust” achievable trajectory, which becomes more robust with increasing field std. 
- With increasing std, *everything* becomes relatively more robust – so while the negative SISUs remain “less robust” than the zero SISU for that training condition, they may be more robust than the zero SISU for a weaker training condition (for example)
- The relationship starts to break down around std 2.0

Std 0.4
![[file-20241126113220236.png]]


Std 1.2
![[file-20241126113300126.png]]

Std 1.6
![[file-20241126113313865.png]]

Std 2.0
![[file-20241126114358068.png]]

###### Comparison of -2, 0, and 2 SISU trajectories, across train stds

![[file-20241126121811342.png]]
![[file-20241126121822102.png]]
![[file-20241126121830822.png]]

###### Same comparison but for disturbance amplitude 2 (instead of 4)

This makes it clearer that 

- hyperrobustness does not occur with this training method, and in fact not much happens when we
  increase the SISU above 1;
- there is significant variation in robustness for different train stds, at SISU 0 (compare this to the PAI method below, where stds up to 1.2 are all pretty similar for SISU 0)

SISU -2
![[file-20241128111504513.png]]
SISU 0
![[file-20241128111521393.png]]
SISU 1
![[file-20241128111543963.png]]
SISU 2
![[file-20241128111610673.png]]

##### Trained with PAI-ASF

- The position trajectories for SISU 0 look reasonably similar to [[results-1#^compare-curl-train-aligned|those]] from part 1

###### Train std. 0.0

If train std is 0.0, there is only a slight difference in behaviour as SISU changes, presumably because the SISU is still driving a small change in network dynamics.
![[file-20241214103416783.png]]
###### Train std 0.5

Even in the absence of curl, SISU -3 shows some unrobust behaviour

![[file-20241214103643630.png]]
Versus at curl 2
![[file-20241214103718104.png]]
###### Train std 1.0
![[file-20241214103844637.png]]
###### Train std 1.5

Note that at higher SISU, the control force starts oscillating, but that this does not appear to affect performance *on average*.
![[file-20241214103938075.png]]
###### Comparison of -2, 0, and 2 SISU trajectories, across train stds

- SISU -2 is the most similar across stds. 
- SISU 0 shows a spread, suggesting the entire network learns to be more robust regardless of SISU

![[file-20241214104101160.png]]
![[file-20241214104116202.png]]
![[file-20241214104135071.png]]
#### Trained and evaluated on curl fields, 4-step delay

**Evaluation curl fields have amplitude 2. Note that this is only half as strong as before, since curl amplitude 4 is very unstable for networks trained on delay 4.**
##### Trained with BCS-75

TODO.
##### Trained with DAI

- As in the non-delayed case, there is a kind of asymptotic effect with the SISU, and the achievable robustness depends on the train std
- Very high values of the SISU can smooth out the oscillations, but do not seem to be able to decrease the lateral deviations beyound some bound, even as the train std is increased to the point that things because absolutely unstable

Std 0.4
![[file-20241126135507506.png]]


Std 0.8
![[file-20241126135524479.png]]

The highest SISU here is particular interesting
![[file-20241126135809600.png]]

Std 1.2
![[file-20241126135617359.png]]

###### Comparison of -2, 0, and 1 SISU trajectories, across train stds

Something weird happens before train std 2.

![[file-20241126143954222.png]]

![[file-20241126144011282.png]]
![[file-20241126144022068.png]]
##### Trained with PAI-ASF

TODO.

###### Comparison of -2, 0, and 1 SISU trajectories, across train stds

TODO.

#### Trained and evaluated on constant fields, no delay

##### Determining the train stds. to compare

This was less obvious to me than with curl fields. 

- The switch to a robust strategy happens at quite low field std, and saturates at stds not much higher. 

##### Trained with BCS-75

###### Train std 0.0
![[file-20241216103201955.png]]
![[file-20241216103214101.png]]
###### Train std. 0.02, 0.04, 0.16
![[file-20241216103238833.png]]
![[file-20241216103250619.png]]
![[file-20241216103301681.png]]
###### Comparison of -2, 0, 2 SISU trajectories, across train stds

-2, No perturbation: 
![[file-20241216103339839.png]]

-2, with perturbation:
![[file-20241216103354503.png]]
0, with:
![[file-20241216103418936.png]]
1, with:
![[file-20241216103443876.png]]
2, with:
![[file-20241216103434342.png]]
##### Trained with PAI-ASF

###### Train std 0.0
![[file-20241216104448977.png]]
![[file-20241216104453579.png]]

###### Train std. 0.02, 0.04, 0.16
![[file-20241216104459447.png]]
![[file-20241216104506681.png]]
![[file-20241216104513355.png]]

###### Comparison of -2, 0, 1, 2 SISU trajectories, across train stds
![[file-20241216104650490.png]]
![[file-20241216104700020.png]]
![[file-20241216104741439.png]]
![[file-20241216104717311.png]]

### Distributions of performance measures

#### Trained and evaluated on curl fields, no delay

- The max forward force is ~identical both with and without evaluation curl. This suggests an initial open-loop difference in strategy due to SISU.

##### BCS-75
###### No disturbance
![[file-20241214110545881.png]]

![[file-20241214110622630.png]]
![[file-20241214110729576.png]]
![[file-20241214110851931.png]]
###### Curl field 2
![[file-20241214110606791.png]]
![[file-20241214110634077.png]]
![[file-20241214110740443.png]]
![[file-20241214110905232.png]]
##### PAI-ASF
###### No disturbance
![[file-20241214104437650.png]]
![[file-20241214104635763.png]]
![[file-20241214104755870.png]]
![[file-20241214104938360.png]]
###### Curl field 2
![[file-20241214104450761.png]]
![[file-20241214104648127.png]]
![[file-20241214104818379.png]]
![[file-20241214104951615.png]]
#### Trained and evaluated on curl fields, 4-step delay

TODO.

## Feedback perturbations

Likewise, show that changing the SISU appears to change the feedback gains.

### Aligned trajectories

These are all for impulse magnitude 1.2 (pos) and 0.8 (vel) unless otherwise stated.
#### Trained on curl fields

##### BCS

###### Position feedback impulse
![[file-20241215103958440.png]]
![[file-20241215104025876.png]]
![[file-20241215104110235.png]]
![[file-20241215104120989.png]]
![[file-20241215104129745.png]]
![[file-20241215104140340.png]]
###### Velocity feedback impulse

Very similar to position.
![[file-20241215104206457.png]]
![[file-20241215104234073.png]]

##### PAI-ASF

###### Position feedback impulse



![[file-20241215095026554.png]]
![[file-20241215094938895.png]]
![[file-20241215095152027.png]]
![[file-20241215095202834.png]]
![[file-20241215095221625.png]]
![[file-20241215095236212.png]]

###### Velocity feedback impulse

Very similar in general to a position perturbation.
![[file-20241215095343039.png]]
![[file-20241215095400390.png]]
#### Trained on constant fields
##### BCS-75

###### Position feedback impulse

**Std 0.0**
![[file-20241215121924602.png]]
![[file-20241215121952032.png]]
![[file-20241215122001866.png]]
![[file-20241215122010029.png]]
![[file-20241215122025009.png]]
![[file-20241215122033905.png]]
**Std 0.04**
![[file-20241215122119793.png]]
![[file-20241215122127390.png]]
![[file-20241215122216595.png]]
![[file-20241215122222624.png]]
![[file-20241215122234995.png]]
![[file-20241215122241437.png]]

###### Velocity feedback impulse

Very similar to the responses for position, except somewhat smaller forces. 

##### DAI

###### Position feedback impulse

**Std 0.0**

![[file-20241201132111309.png]]
The SISUs are slightly more separated here than they were for BCS, but only slightly.

![[file-20241201132142649.png]]

**Std 0.04**
![[file-20241201132252629.png]]
The relationship in the orthogonal forces seems more ordered than in BCS, which is probably related to the extremely bounded (i.e. no hyper-robust) performance seen for DAI. Not that SISU 2 does not oscillate much more than the others.

![[file-20241201132332929.png]]
The same orderliness is also reflected in the positions:
![[file-20241201132534688.png]]

These are a lot more similar than they are for BCS. There is higher variance for SISU -2, but it is not much worse than any of the other SISUs. 
![[file-20241201132638777.png]]

###### Velocity feedback impulse

**Std 0.04**
![[file-20241201132307483.png]]

##### PAI-ASF
###### Position feedback impulse

**Std 0.0**
![[file-20241216095109021.png]]
![[file-20241216095256930.png]]
![[file-20241216095413835.png]]
![[file-20241216095441212.png]]
![[file-20241216095459275.png]]
![[file-20241216095524679.png]]
**Std 0.04**:
![[file-20241216095124711.png]]
![[file-20241216095328537.png]]
![[file-20241216095420583.png]]
![[file-20241216095446629.png]]
![[file-20241216095507718.png]]
![[file-20241216095548120.png]]
###### Velocity feedback impulse

Again, usually very similar to position feedback impulse response.

### Distributions of performance measures

Note that the grey violins are the std 0.0 condition, which is being compared with one of the other conditions (1.0 amp curl, or 0.04 amp constant)

- Baseline (train std 0.0) still sees some variation between SISUs — often at negative SISU. This is OK since the SISU drives a network even that has not optimized to use it for anything, especially outside the range (0-1) that it was trained on. 
- [ ] When publishing these, generate the grey/baseline sets as separate figs, use `get_underlay_fig` to remove their axes etc, export to images, and then composite them as images

#### Trained on curl fields

##### BCS-75
![[file-20241215104309957.png]]
![[file-20241216101436318.png]]
![[file-20241216101448722.png]]
![[file-20241216101651542.png]]
Here the difference between the pos and vel impulses is clearer… notice the larger difference between baseline (std 0.0) and disturbance train conditions, and the larger and more variable overall responses.
![[file-20241216101708851.png]]
![[file-20241216101813906.png]]
![[file-20241216101823561.png]]
![[file-20241216101511881.png]]
![[file-20241216101517348.png]]
![[file-20241216101535260.png]]
![[file-20241216101542040.png]]
![[file-20241216101632362.png]]
![[file-20241216101638038.png]]
##### PAI-ASF
![[file-20241215095719417.png]]
![[file-20241216102122748.png]]
![[file-20241216102130518.png]]
![[file-20241216102350079.png]]
![[file-20241216102356117.png]]
![[file-20241216102424795.png]]
![[file-20241216102434895.png]]
![[file-20241216102236150.png]]
![[file-20241216102242150.png]]
![[file-20241216102253571.png]]
![[file-20241216102258494.png]]
![[file-20241216102321644.png]]
![[file-20241216102327569.png]]
#### Trained on constant fields

##### BCS-75
![[file-20241215122459813.png]]
![[file-20241216100642301.png]]
![[file-20241216100705122.png]]
![[file-20241216100848805.png]]
![[file-20241216100854446.png]]
![[file-20241216100906024.png]]
![[file-20241216100911461.png]]
![[file-20241216100716511.png]]
![[file-20241216100723164.png]]
![[file-20241216100744191.png]]
![[file-20241216100759562.png]]
![[file-20241216100829474.png]]
![[file-20241216100835107.png]]

##### PAI-ASF
![[file-20241216095648472.png]]
![[file-20241216095655928.png]]
![[file-20241216100004803.png]]
![[file-20241216100010822.png]]
![[file-20241216100026890.png]]
![[file-20241216100036784.png]]
![[file-20241216095757477.png]]
![[file-20241216095807829.png]]
![[file-20241216095816799.png]]
![[file-20241216095823957.png]]
![[file-20241216095930687.png]]
![[file-20241216095937157.png]]

## Dynamics

### Initial and final fixed points

These FPs are for a set of center-out reaches. 

- The goals-goals (steady state) FPs here correspond to the network being at rest, at the reach endpoint. 
- The inits-goals FPs are the FPs on the very first timestep, when the network’s inputs tell it it’s at the origin but that its target is one of the center-out positions
- Trajectories of FPs correspond to the FPs of the network for each of the inputs it actually had, during the reaches
#### Trained on curl fields, evaluated on baseline
##### DAI

##### PAI
###### SISU 0, comparing 4 reach directions
![[plotly-20241212-141528]]

###### SISU 1, comparing 4 reach directions
![[plotly-20241212-141647]]

###### Comparison of SISUs for a single direction
![[plotly-20241212-142259]]

#### Trained on curl fields, evaluated on amp. 2 curl field
##### DAI
###### SISU 0, comparing 4 reach directions
![[plotly-20241212-134054]]

![[plotly-20241212-134110]]

###### Comparing SISUs, for a single reach direction
![[plotly-20241212-134208]]

###### Goals-goals and inits-goals FPs across SISUs
![[plotly-20241212-134339]]

##### PAI

###### SISU 0, comparing 4 reach directions

![[plotly-20241212-132341]]
![[plotly-20241212-132845]]

###### SISU 1, comparing 4 reach directions
![[plotly-20241212-132631]]
![[plotly-20241212-132812]]


###### Comparison of SISUs for a single direction
![[plotly-20241212-133032]]
###### Goals-goals and inits-goals FPs across SISUs
![[plotly-20241212-133252]]


### Eigendecomposition of steady-state FP Jacobians

This is for a grid of steady (i.e. goal-goal) state positions across the workspace. 

#### Trained on curl fields

##### DAI
###### Std 0.4
![[plotly-20241211-153915]]

###### Std 1.2
![[plotly-20241211-154004]]

##### PAI

###### Std 0.4
![[plotly-20241211-154150]]

###### Std 1.2
![[plotly-20241211-154033]]







## Other/supplementary analyses

### Effect of delay on robustness scaling

As we increase the feedback delay, how does the relationship between X and the SISU change?

Be decisive about what X is.

### Output correlations

The readout norm will be fixed for each hybrid network; however, it may be the case that the output correlation (i.e. partitioning of activity between null and potent spaces) will vary with the SISU. 

Quantify this.

### Floor on control forces? Minimum work needed to complete task?

This is for the zero-noise, curl amplitude 2 condition, trained on curls. Note that as we increase train std and SISU, the integral over the absolute lateral forces bottoms out around 20. Is this because there is a minimum amount of work that is necessary to complete the task, under these conditions, which we are approaching by adopting a more robust strategy?

![[file-20241128121329199.png]]