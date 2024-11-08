---
created: 2024-09-24T10:14
updated: 2024-11-07T17:21
---
## Training networks on different disturbance levels

### No fields

#### No noise, no delay
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/1-1__loss-history__curl__std-0__replicates-10.png]]
#### 0.04 noise, no delay
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/1-1__loss-history__random__std-0__replicates-10.png]]
Clearly noise affects the balance of the loss terms, in particular it puts a floor on the final velocity error (makes sense since due to the motor noise), and likewise it also increases the position error a bit.
#### 0.1 noise, no delay

More noise → higher floor on the velocity and position errors, but qualitatively it looks like the overall evolution is similar, in particular considering the hidden and control force errors remaining so similar in evolution.
![[1-1__loss-history__curl__std-0__replicates-10 1.png]]
#### Zero noise, 2 step delay

This looks almost identical to the zero noise, zero delay case. Presumably, in the absence of unpredictable disturbances, the learning process is essentially identical.
![[1-1__loss-history__curl__std-0__replicates-10 2.png]]
#### Zero noise, 4 step delay

Again, almost identical to the zero noise, zero delay case.
![[1-1__loss-history__random__std-0__replicates-10 1.png]]
#### 0.1 noise, 4 step delay

TODO

### Curl fields

#### Zero noise, no delay

Curl field std. 0.8
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/1-1__loss-history__curl__std-0.8__replicates-10.png]]
It looks like the most interesting policy developments might happen between iterations 10 and 100.

Curl field std. 1.6
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/1-1__loss-history__curl__std-1.6__replicates-10.png]]
Curl field std. 2.4. At this level, the fields appear to be too strong for the models to converge, at least on average.
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/1-1__loss-history__curl__std-2.4__replicates-10.png]]
It is clear from the loss distribution over replicates that everything is more or less fine except at the highest curl std:
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/best-loss-distn-by-replicate.png]]
#### 0.04 noise, no delay

Curl std. 0.8. 
![[1-1__loss-history__curl__std-0.8__replicates-10 1.png]]
Curl std. 1.6

![[1-1__loss-history__curl__std-1.6__replicates-10 1.png]]
Curl std. 2.4
![[1-1__loss-history__curl__std-2.4__replicates-10 1.png]]
![[best-loss-distn-by-replicate 1.png]]
#### 0.1 noise, no delay

Curl std 0.8
![[1-1__loss-history__curl__std-0.8__replicates-10 2.png]]
Curl std. 1.6
![[1-1__loss-history__curl__std-1.6__replicates-10 2.png]]
Curl std. 2.4
![[1-1__loss-history__curl__std-2.4__replicates-10 2.png]]
Again, qualitatively this is as expected given the lower-noise conditions.
![[best-loss-distn-by-replicate 2.png]]
#### Zero noise, 2 step delay

Curl std. 0.8. Doesn’t look very different from the zero noise, zero delay case.
![[1-1__loss-history__curl__std-0.8__replicates-10 3.png]]
Curl std. 1.6. Now things are looking different from the zero noise, zero delay case. Clearly some or all of the replicates are not stable across training. However, notice that the state errors initially decrease before increasing again.
![[1-1__loss-history__curl__std-1.6__replicates-10 3.png]]
Curl std. 2.4. Apparent divergence. In particular, the hidden loss appears to be hitting a ceiling, which suggests the tanh units are saturating.
![[1-1__loss-history__curl__std-2.4__replicates-10 3.png]]

Comparing the losses across replicates on the final iteration versus the best iteration (for each replicate), it is easy to see that the std. 2.4 models simply diverge, whereas the std. 1.6 models reach a low-ish loss at some point during training.
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/final-loss-distn-by-replicate.png]]
![[best-loss-distn-by-replicate 3.png]]
#### Zero noise, 4 step delay

Curl std. 0.8. Some minor signs of instability towards the end.
![[1-1__loss-history__curl__std-0.8__replicates-10 4.png]]
Curl std. 1.6. More pronounced and definitive divergence than the equivalent condition in the 2-step delay case.
![[1-1__loss-history__curl__std-1.6__replicates-10 4.png]]
The std. 2.4 case is as expected from the 2-step delay case.

### Random constant fields

#### Zero noise, zero delay

Std. 0.01. Very similar to the no-fields case.
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/1-1__loss-history__random__std-0.01__replicates-10.png]]
Std. 0.1. Some qualitative changes happening in the early period.
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/1-1__loss-history__random__std-0.1__replicates-10.png]]
Std. 1.0. Even more pronounced changes in the early period. 
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/1-1__loss-history__random__std-1.0__replicates-10.png]]
Even at the highest training std., the total loss is small. Presumably if we increased the field strength enough, then it would start to trade off harder with weight decay in the output layer in order to maintain sufficient steady state controlf force. I don’t think there’s a hard ceiling on the control forces though?![[best-loss-distn-by-replicate 4.png]]
#### 0.1 noise, zero delay

Std. 1.0. Mostly just puts a floor on the effector errors, as expected. 
![[1-1__loss-history__random__std-1.0__replicates-10 1.png]]
Systematic increase in the loss with the std, though overall the values are small.
![[best-loss-distn-by-replicate 5.png]]

#### Zero noise, 4 steps delay

Std. 1.0. The delay is not nearly as problematic as it was for the curl field, certainly because the field here is constant and does not interact with the policy in a time-delayed way.
![[1-1__loss-history__random__std-1.0__replicates-10 2.png]]
Std. 2.0. 
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/1-1__loss-history__random__std-2.0__replicates-10.png]]
![[best-loss-distn-by-replicate 6.png]]
## Example center-out sets 

These show a single evaluation of the replicate which had the lowest total loss on the respective training condition. 

I also generated plots that show the variance over the replicates, but I will only include these in this document in a couple 
### No noise, no delay
#### No perturbation

##### No training perturbation
![[curl-amp-0.0__curl-train-std-0__rep-6__eval-0.png]]

###### All replicates  %% fold %%
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/curl-field-0.0__curl-train-std-0__eval-0.png]]
##### Trained on curl fields
![[curl-amp-0.0__curl-train-std-1.6__rep-3__eval-0.png]]
###### All replicates  %% fold %%

Interesting that the variance in the control forces is a bit higher between replicates than it was for the control network. Note that the control forces are larger, but that this is the no-noise condition, so this effect cannot be due to multiplicative noise.
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/curl-field-0.0__curl-train-std-1.6__eval-0.png]]
##### Trained on random constant fields
![[random-amp-0.0__random-train-std-1.0__rep-0__eval-0.png]]
###### All replicates  %% fold %%

The variance in the control forces is even greater than for networks trained on curl fields. Again, this has nothing to do with noise.
![[random-field-0.0__random-train-std-1.0__eval-0.png]]
#### Curl field perturbation
##### No training perturbation
![[curl-amp-4.0__curl-train-std-0__rep-6__eval-0.png]]
###### All replicates  %% fold %%
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/curl-field-4.0__curl-train-std-0__eval-0.png]]
###### Evaluated with system noise  %% fold %%

Whatever policy the network learned, its performance appears not to be significantly affected by system noise. Note that the following plot shows multiple evaluations for a single replicate.
![[curl-field-4.0__curl-std-0__replicate-6.png]]
##### Trained on curl fields
![[curl-amp-4.0__curl-train-std-1.6__rep-3__eval-0.png]]
###### All replicates  %% fold %%
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/curl-field-4.0__curl-train-std-1.6__eval-0.png]]

###### Evaluated with system noise  %% fold %%

Similarly to the control case, noise does not make the policy ineffective.
![[curl-field-4.0__curl-std-1.6__replicate-3.png]]
##### Trained on random constant fields

- This is weird. 
- I suppose it is because the attractors that get strengthened by training on random fields are the ones that output a constant force at the target, but because the network is not accustomed to the curl, it tries to correct the error (similarly but a little better than the control network) until it approaches a ring of constant-force attractors that allow it to orbit the target.
- [ ] Try running this eval for twice as long (200 steps) and see if it keeps orbiting or if it become unstable.
![[curl-amp-4.0__random-train-std-1.0__rep-0__eval-0.png]]
###### All replicates  %% fold %%
![[curl-field-4.0__random-train-std-1.0__eval-0.png]]
###### Evaluated with system noise  %% fold %%

- Noise perhaps has a slightly worse negative effect than it did in the control and curl-trained conditions.
- Probably because the constant-force orbit attractors are sensitive to changes in position.  
- [ ] Also try this one for 200 steps?
![[curl-field-4.0__random-std-1.0__replicate-0.png]]
#### Random constant field perturbation
##### No training perturbation

- [ ] Perhaps the perturbation strengths should be bit higher
- Is there any field strength at which the system will be unstable? Probably not, due to the lack of feedback with the disturbance.
![[random-amp-0.4__random-train-std-0__rep-6__eval-0.png]]

###### All replicates  %% fold %%
![[random-field-0.4__random-train-std-0__eval-0.png]]
###### Evaluated with system noise  %% fold %%

##### Trained on random constant fields
![[random-amp-0.4__random-train-std-1.0__rep-0__eval-0.png]]
###### All replicates  %% fold %%
![[random-field-0.4__random-train-std-1.0__eval-0.png]]
###### Evaluated with system noise  %% fold %%
![[random-field-0.4__random-std-1.0__replicate-0.png]]
##### Trained on curl fields

- This fares better than the opposite case, where we trained on random fields and evaluated on curl fields. 
- The network is able to reduce most of the deviation caused by the field, and stop very close to the target.
![[random-amp-0.4__curl-train-std-1.6__rep-3__eval-0.png]]
###### All replicates  %% fold %%
![[random-field-0.4__curl-train-std-1.6__eval-0.png]]
###### Evaluated with system noise  %% fold %%
![[random-field-0.4__curl-std-1.6__replicate-3.png]]
### No noise, but with delay
#### No perturbation

##### No training perturbation

A **2-step delay** has almost no effect, presumably because in the absence of perturbations it is straightforward to simply offset the policy by the appropriate number of steps. 
![[curl-field-0.0__curl-train-std-0__eval-0 1.png]]

Likewise, increasing the delay to **4 steps** also has very little effect.
![[curl-field-0.0__curl-train-std-0__eval-0 2.png]]
##### Trained on curl fields

The network can still complete the task at **2 steps delay**, but the variance between replicates is much higher, presumably because curl fields are a delay-sensitive perturbation (see below).
![[curl-field-0.0__curl-train-std-1.6__eval-0 1.png]]
And a **4-step delay**
![[curl-field-0.0__curl-train-std-1.6__eval-0 2.png]]
##### Trained on random constant fields

###### 4-step delay  %% fold %%
![[random-field-0.0__random-train-std-1.0__eval-0 1.png]]

- More variable than the control network, but not so bad as the curl one

#### Curl field perturbation
##### No training perturbation

###### 2-step delay  %% fold %%
![[curl-field-4.0__curl-train-std-0__eval-0 2.png]]

- The rate and severity of the “curl oscillations” is increased by the delay.

###### 4-step delay  %% fold %%
![[curl-field-4.0__curl-train-std-0__eval-0 1.png]]

- The control network is totally unstable in this setting. 
- If we look at the unit activities, they are probably saturating, trying to compensate for movements of the effector away from the intended direction, which it always learns about too late to do anything.

##### Trained on curl fields
###### 2-step delay  %% fold %%
![[curl-field-4.0__curl-train-std-1.6__eval-0 1.png]]
###### 4-step delay  %% fold %%
![[curl-field-4.0__curl-train-std-1.6__eval-0 2.png]]

- Clearly, delay negatively impacts the ability to deal with curl fields
- Presumably: by the time the network receives feedback, the curl has already bent the effect of the network’s previous control away from the intended direction.
- However, the network has to try to up some of its gains to be robust to the perturbations, and this can lead to 
- The network is still better than the control network
- It is interesting that some of the replicates are “loopy”, but these loops are tighter than the control network’s, probably because the control gains are higher.

##### Trained on random constant fields
###### 2-step delay  %% fold %%
![[curl-field-4.0__random-train-std-1.0__eval-0 1.png]]

- This is the condition that had the orbits, prior to the delay.
- The addition of a short delay makes this condition unstable.

###### 4-step delay  %% fold %%
![[curl-field-4.0__random-train-std-1.0__eval-0 2.png]]

- Total instability/saturation in this condition.

#### Random constant field perturbation

##### No training perturbation
###### 4-step delay  %% fold %%
![[random-field-0.4__random-train-std-0__eval-0 1.png]]

- As expected (see below) the delay does not seriously affect the control network’s policy, which ends up being influenced by the constant field more or less like it would have been without the delay. 
- In other words, this condition is to the undelayed control network perturbed by random field, what the delayed unperturbed control network is to the undelayed unperturbed control network.
- Neither the delay nor the field creates an unstable feedback in the system.
##### Trained on curl fields
###### 2-step delay  %% fold %%
![[random-field-0.4__curl-train-std-1.6__eval-0 1.png]]
###### 4-step delay  %% fold %%
![[random-field-0.4__curl-train-std-1.6__eval-0 2.png]]

- Performance isn’t great, probably because the network was trained in a condition where it was nearly unstable, so the resulting policy is not great at reaching the goal in any case.
- On the other hand, performance in the 2-step case just above was okay, because the network was still able to perform pretty well in its training condition.

##### Trained on random constant fields
###### 2-step delay  %% fold %%
![[random-field-0.4__random-train-std-1.0__eval-0 1.png]]
###### 4-step delay  %% fold %%
![[random-field-0.4__random-train-std-1.0__eval-0 2.png]]


- Even with a 4-step delay, the performance of the policy learned for compensating for random constant fields is not degraded.
- Presumably this is because there is not a time-sensitive interaction between network actions and “field actions”, like there was for the curl field, given that the field is constant in this case. 

### 0.1 noise, no delay (TODO)

- [ ] Compile some figures suggesting that training on noise doesn’t significantly alter the resulting policies
- However, it might somewhat decrease performance on noise-free conditions? Since the network was trained with a floor on its loss, and thus may not be adapted to minimize state errors below a certain point?

## Aligned trajectories
### No noise, no delay
#### Evaluated on curl fields
##### No training perturbation
![[curl-train-std-0.png]]
##### Trained on max curl field std.
![[curl-train-std-1.6.png]]
##### Comparison across curl training conditions
![[curl-field-4.0.png]]

##### Trained on max random field std.
![[random-train-std-1.0.png]]

- Here the “orbit” feature develops after the field strength exceeds a certain level.

##### Comparison across random constant field training conditions
![[curl-field-4.0 1.png]]

But also consider a test curl of a small amplitude:

![[curl-field-1.0.png]]

Here the “loop” at the end is much smaller and does not develop into an orbit, and the robustness advantage of the networks trained on random constant fields is a bit clearer. 

#### Evaluated on random constant fields
##### No training perturbation
![[random-train-std-0.png]]
##### Trained on max. random constant fields
![[random-train-std-1.0 1.png]]
##### Comparison across random constant field training conditions
![[random-field-0.4.png]]
##### Trained on max. curl fields
![[curl-train-std-1.6 1.png]]
##### Comparison across curl field training conditions
![[random-field-0.4 1.png]]

- Curiously, it tilts slightly away from the goal 

### No noise, with delay

#### Evaluated on curl fields
##### No training perturbation
###### 2-step delay %% fold %%
![[curl-train-std-0 1.png]]
###### 4-step delay %% fold %%
![[curl-train-std-0 2.png]]

##### Trained on max. curl fields
###### 2-step delay %% fold %%
![[curl-train-std-1.6 2.png]]

Compare this to the second-highest curl field training condition:
![[curl-train-std-0.8.png]]

Notice that the solution is not quite as robust, but also the replicates are much less variable. 
###### 4-step delay %% fold %%
![[curl-train-std-1.6 3.png]]

- Interesting in this case that in the absence of perturbations, the average trajectory appears slightly in the opposite direction that the perturbation would have pushed it. 
- Of course this cannot be a reaction to the perturbation direction, since the networks evaluated without perturbations are totally naive to the perturbation direction. 
- So this is probably just a slight bias due to variance between the replicates.

And the next-strongest condition:
![[curl-train-std-0.8 1.png]]
##### Comparison across curl field training conditions
###### 2-step delay %% fold %%
![[curl-field-4.0 2.png]]

And on the next-weaker evaluation curl:
![[curl-field-2.0.png]]

###### 4-step delay %% fold %%
![[curl-field-4.0 3.png]]

- It’s difficult to see here, but the two highest train field stds. are stable or near-stable.

Looking at the next-weaker evaluation curl:
![[curl-field-2.0 1.png]]

This is easier to compare with the 2-step delay case just above.

##### Trained on max. random constant fields
###### 2-step delay %% fold %%
![[random-train-std-1.0 2.png]]

- Clearly the delay destabilizes this policy at higher curl field strengths

###### 4-step delay %% fold %%

Here’s the second-strongest eval condition:
![[random-train-std-0.1.png]]

The strongest one isn’t very meaningful to see here since the pink curve is so unstable that nothing else is visible.
##### Comparison across random constant field training conditions
###### 2-step delay %% fold %%
![[curl-field-4.0 5.png]]

So training on random constant fields makes things worse due to how vigorous the control is.

Here’s the next-weakest eval condition:
![[curl-field-2.0 2.png]]
###### 4-step delay %% fold %%
![[curl-field-4.0 6.png]]

![[curl-field-2.0 3.png]]
#### Evaluated on random constant fields
##### No training perturbation
###### 2-step delay %% fold %%
![[random-train-std-0 1.png]]
###### 4-step delay %% fold %%
![[random-train-std-0 2.png]]

- This is slightly worse than the 2-step case. 
- [ ] This is interesting and maybe says something about why the control network has difficulty getting to the goal position even in the absence of delay.
##### Trained on max. random constant fields
###### 2-step delay %% fold %%
![[random-train-std-1.0 3.png]]
###### 4-step delay %% fold %%
![[random-train-std-1.0 4.png]]

- The shape of the control forces is a bit different, but overall this isn’t nearly as affected by the delay as the curl field was.
##### Comparison across random constant field training conditions
###### 2-step delay %% fold %%
![[random-field-0.4 2.png]]
###### 4-step delay %% fold %%
![[random-field-0.4 3.png]]
##### Trained on max. curl fields
###### 2-step delay %% fold %%
![[curl-train-std-1.6 4.png]]

And the next-weakest training condition:

![[curl-train-std-0.8 2.png]]

- As expected, weird things happen when we try to use a network trained on curl+delay for other tasks, since it was trained in a fundamentally confusing condition.
- This is even more pronounced in the 4-step delay, below.
###### 4-step delay %% fold %%
![[curl-train-std-1.6 5.png]]

![[curl-train-std-0.8 3.png]]

##### Comparison across curl field training conditions
###### 2-step delay %% fold %%
![[random-field-0.4 4.png]]

- So training on curl+delay is still more robust than training the control network with delay; however the delay increases the endpoint error. 
- [ ] This may be worth looking at again.
###### 4-step delay %% fold %%
![[random-field-0.4 5.png]]
- The “kick” away from the goal is a bit exaggerated in the stronger training conditions, here.
### 0.1 noise, no delay

#### Evaluated on curl fields
##### No training perturbation versus max curl field training perturbation
![[curl-train-std-0 3.png]]
![[curl-train-std-1.6 6.png]]

- This is pretty similar to training without noise, except of course that the average force trajectories [[results-1#^xe3mdu]]
- [ ] I haven’t yet evaluated a model trained on noise, in the absence of noise. This would reveal if the control forces/policy really are different due to training on noise.

##### Comparison of curl field training conditions
![[curl-field-4.0 7.png]]

- One thing that looks a bit different here is that the “elbows” are less evident in the mean trajectory for the zero curl training condition. 
- However, looking at the individual curves, there are definitely elbows; it is probably that the noise causes them to happen at somewhat different times and this gets averaged out. 

##### Comparison of random constant field training conditions

- [ ] I haven’t evaluated noise-trained random constant field models on noise+curl yet.

#### Evaluated on random constant fields
#####  No training perturbation versus max. random constant field training perturbation
![[random-train-std-0 3.png]]
![[random-train-std-1.0 5.png]]

- As with the curl field, the resulting trajectories are similar to the no-noise condition
- Interesting that the little crook away from the goal is present for the zero training std network; this was also seen when using curl-trained networks to mitigate random constant fields. 
##### Comparison of random constant field training conditions
![[random-field-0.4 6.png]]

## Velocity profiles
### No noise, no delay
#### No perturbation
##### Comparison of curl field training conditions
![[curl-field-0.0_forward-vels.png]]

- One thing that is clear is that training on stronger curl fields leads to higher acceleration at the beginning of the reach.
- However, it does not necessarily lead to higher peak velocity in this case.
- This makes sense given that higher peak velocity leads to stronger curl.
- [ ] Perhaps in the delay condition we should see the peak velocity decrease when trained on stronger curl fields?

![[curl-field-0.0_lateral-vels.png]]

- These values are overall very low
- But the variation in lateral velocity is higher for stronger training conditions, which is interesting.

##### Comparison of random constant field training conditions
![[random-field-0.0_forward-vels.png]]

- The initial acceleration also increases with training std., as with the curl fields
- But unlike curl training, the peak velocity also increases, which makes sense – the disturbance is not aggravated by velocity, here.
- Thus this comparison looks more like what we expect from a robust controller.

![[random-field-0.0_lateral-vels.png]]
#### Evaluated on curl fields
##### Comparison of curl field training conditions
![[curl-field-4.0_forward-vels.png]]

- Here the peak velocity definitely increases with train curl std.
- [ ] Look at the velocity profile for the best replicate at train field std. 2.4: does the “double peak” disappear?

![[curl-field-4.0_lateral-vels.png]]

Looking at the next-weakest curl test condition shows that the strength of the effect on the forward vs. lateral velocity profile depends on curl strength.
![[curl-field-2.0_forward-vels.png]]
![[curl-field-2.0_lateral-vels.png]]
##### Comparison of random constant field training conditions
![[curl-field-4.0_forward-vels 2.png]]
![[curl-field-4.0_lateral-vels 2.png]]

The “orbiting” is clear here. What about the next-weakest curl strength?
![[curl-field-2.0_forward-vels 2.png]]
![[curl-field-2.0_lateral-vels 2.png]]
#### Evaluated on random constant fields
##### Comparison of curl field training conditions
![[random-field-0.4_forward-vels.png]]

- Interestingly the forward velocity looks almost identical to what it did when trained on curl fields and then tested without perturbation.

![[random-field-0.4_lateral-vels.png]]

##### Comparison of random constant field training conditions
![[random-field-0.4_forward-vels 1.png]]

- Almost identical to the no-perturbation condition, when trained on random constant fields. 

![[random-field-0.4_lateral-vels 1.png]]

### No noise, 2-step delay
#### No perturbation
##### Comparison of curl field training conditions
![[curl-field-0.0_forward-vels 1.png]]
![[curl-field-0.0_lateral-vels 1.png]]

##### Comparison of random constant field training conditions
![[random-field-0.0_forward-vels 1.png]]

- Very similar to the no-delay condition.

![[random-field-0.0_lateral-vels 1.png]]
#### Evaluated on curl fields
##### Comparison of curl field training conditions
![[curl-field-4.0_forward-vels 1.png]]
![[curl-field-4.0_lateral-vels 1.png]]

And for the next-weakest eval curl:
![[curl-field-2.0_forward-vels 1.png]]
![[curl-field-2.0_lateral-vels 1.png]]

##### Comparison of random constant field training conditions
![[curl-field-4.0_forward-vels 3.png]]
![[curl-field-4.0_lateral-vels 3.png]]

The more vigorous response of the networks trained on random fields are counterproductive when there is curl+delay.

#### Evaluated on random constant fields
##### Comparison of curl field training conditions
![[random-field-0.4_forward-vels 2.png]]
![[random-field-0.4_lateral-vels 2.png]]

- Keeping with the trend seen previously, the forward velocities are very similar to the no-perturbation test condition for these models.
- The variance of the responses is significantly higher for the 1.6 std train condition, presumably because this level of curl+delay during training makes it difficult to learn a coherent policy.

##### Comparison of random constant field training conditions
![[random-field-0.4_forward-vels 3.png]]

![[random-field-0.4_lateral-vels 3.png]]

- Both the forward and lateral profiles are similar to the no-delay case.

### 0.1 noise, no delay (TODO)
#### No perturbation
##### Comparison of curl field training conditions
##### Comparison of random constant field training conditions
#### Evaluated on curl fields
##### Comparison of curl field training conditions
##### Comparison of random constant field training conditions
#### Evaluated on random constant fields
##### Comparison of curl field training conditions
##### Comparison of random constant field training conditions


## Distributions of performance measures

Here I’ll show the full plots with all train and test conditions; however I have also produced figures that only show the 2x2 comparison between the lowest and highest of each condition.

### No noise, no delay (unless otherwise stated)
#### Max forward velocity
##### Train curl, test curl
![[vel-forward-max.png]]
###### 2-step delay %% fold %%
![[vel-forward-max 4.png]]
##### Train random, test random
![[vel-forward-max 1.png]]
###### 2-step delay %% fold %%
![[vel-forward-max 6.png]]
##### Train curl, test random
![[vel-forward-max 3.png]]
##### Train random, test curl
![[vel-forward-max 2.png]]
###### 2-step delay %% fold %%
![[vel-forward-max 5.png]]
#### Max forward control force

- These are always the ~same across field amplitudes, which suggests this is entirely due to the difference in initial acceleration induced by training on perturbations.
- [ ] It might also make sense to look at the sum of forward control forces.

##### Train curl, test curl
![[force-forward-max.png]]

###### 2-step delay %% fold %%
![[force-forward-max 4.png]]
##### Train random, test random
![[force-forward-max 1.png]]
###### 2-step delay %% fold %%
![[force-forward-max 5.png]]
##### Train curl, test random
![[force-forward-max 3.png]]
##### Train random, test curl
![[force-forward-max 2.png]]
###### 2-step delay %% fold %%
![[force-forward-max 6.png]]

- The outlier distribution is due to instability, and not due to a change in the initial accelerating force.

#### Max lateral velocity (in disturbance direction)

- [ ] The max lateral velocity in the opposite direction may also be interesting.

##### Train curl, test curl
![[vel-lateral-left-max.png]]
###### 2-step delay %% fold %%
![[vel-lateral-left-max 4.png]]
##### Train random, test random
![[vel-lateral-left-max 1.png]]
###### 2-step delay %% fold %%
![[vel-lateral-left-max 6.png]]
##### Train curl, test random
![[vel-lateral-left-max 3.png]]
##### Train random, test curl
![[vel-lateral-left-max 2.png]]
###### 2-step delay %% fold %%
![[vel-lateral-left-max 5.png]]
#### Max lateral displacement (in disturbance direction)
##### Train curl, test curl
![[dist-lateral-max.png]]

###### 2-step delay %% fold %%
![[dist-lateral-max 4.png]]
##### Train random, test random
![[dist-lateral-max 1.png]]
###### 2-step delay %% fold %%
![[dist-lateral-max 5.png]]
##### Train curl, test random
![[dist-lateral-max 3.png]]

##### Train random, test curl
![[dist-lateral-max 2.png]]
###### 2-step delay %% fold %%
![[dist-lateral-max 6.png]]
#### Sum of absolute lateral displacements
##### Train curl, test curl
![[dist-lateral-sum.png]]
###### 2-step delay %% fold %%
![[dist-lateral-sum 4.png]]
##### Train random, test random
![[dist-lateral-sum 1.png]]
###### 2-step delay %% fold %%
![[dist-lateral-sum 6.png]]
##### Train curl, test random
![[dist-lateral-sum 3.png]]
##### Train random, test curl
![[dist-lateral-sum 2.png]]
###### 2-step delay %% fold %%
![[dist-lateral-sum 5.png]]
#### Max lateral control force (against disturbance)
##### Train curl, test curl
![[force-counterfield-lateral-max.png]]
###### 2-step delay %% fold %%
![[force-counterfield-lateral-max 4.png]]
##### Train random, test random
![[force-counterfield-lateral-max 1.png]]
###### 2-step delay %% fold %%
![[force-counterfield-lateral-max 5.png]]
##### Train curl, test random
![[force-counterfield-lateral-max 3.png]]
##### Train random, test curl
![[force-counterfield-lateral-max 2.png]]
###### 2-step delay %% fold %%
![[force-counterfield-lateral-max 6.png]]
#### Sum net control forces
##### Train curl, test curl
![[force-net-sum.png]]
###### 2-step delay %% fold %%
![[force-net-sum 4.png]]
##### Train random, test random
![[force-net-sum 1.png]]
###### 2-step delay %% fold %%
![[force-net-sum 6.png]]
##### Train curl, test random
![[force-net-sum 3.png]]
##### Train random, test curl
![[force-net-sum 2.png]]
###### 2-step delay %% fold %%
![[force-net-sum 5.png]]
#### End position error
##### Train curl, test curl
![[error-end-pos.png]]
###### 2-step delay %% fold %%
![[error-end-pos 4.png]]
##### Train random, test random
![[error-end-pos 1.png]]
###### 2-step delay %% fold %%
![[error-end-pos 5.png]]
##### Train curl, test random
![[error-end-pos 3.png]]
##### Train random, test curl
![[error-end-pos 2.png]]

- This is the “orbit” condition, which is why the error is so high for the max train field std.
###### 2-step delay %% fold %%
![[error-end-pos 6.png]]
#### End velocity error
##### Train curl, test curl
![[error-end-vel.png]]
###### 2-step delay %% fold %%
![[error-end-vel 4.png]]
##### Train random, test random
![[error-end-vel 1.png]]
###### 2-step delay %% fold %%
![[error-end-vel 6.png]]
##### Train curl, test random
![[error-end-vel 3.png]]

- Training on curl makes the model very good at stopping, at least.

##### Train random, test curl
![[error-end-vel 2.png]]

- The orbit is relatively high velocity, hence the increase in this error relative to baseline.

###### 2-step delay %% fold %%
![[error-end-vel 5.png]]
### Comparison of delay conditions (TODO)

How much does delay degrade learned performance? 

- [ ] Full 3x3 (control, curl, random) test-train comparison for the zero-noise condition
	- first option: train condition on x axis, eval condition in legend, and zero vs. 4 step in splits

### Comparison of noise conditions (TODO)

How does addition of system noise affect major performance measures (endpoint error, max control?) for different train-test conditions?

### Comparison of noise+delay interactions (TODO?)

e.g. does training on delay increase sensitivity to added noise?

## Feedback perturbations 

### Training on perturbations increases control gain 

In particular, on velocity feedback.
#### Trained on curl fields

To we observe similar effects when training on random constant fields?
##### Max control force during the perturbation
![[force-net-max-during-pert__pos-pert 1.png]]

![[force-net-max-during-pert__vel-pert 1.png]]
##### Max control force after the perturbation
![[force-net-max-after-pert__pos-pert 1.png]]
![[force-net-max-after-pert__vel-pert 1.png]]

##### Comparison of force profiles
###### For a velocity feedback perturbation

![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/F-parallel__pert-amp-1.2000000476837158.png]]
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/F-orthogonal__pert-amp-1.2000000476837158.png]]
- Here it is also clear that the control gains are higher after training on perturbations. 
- However, the lateral controls become much more variable. 
##### Comparison of force profiles during perturbation for position versus velocity feedback impulses

Here, dashed lines are positive feedback impulses, and solid lines are velocity. 

For relatively strong impulses (pos. 1.8 and vel. 1.2), the force profiles are more similar for impulse to position than velocity. 

![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/F-parallel__pert-amp-pos-1.80-vel-1.20__detail.png]]

This effect is less pronounced for weaker impulses (0.6 and 0.4):

![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/F-parallel__pert-amp-pos-0.60-vel-0.40__detail.png]]
###### Comparison of velocity profiles

Here are the equivalent details of the velocity profiles:

![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/v-parallel__pert-amp-pos-1.80-vel-1.20__detail.png]]

![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/v-parallel__pert-amp-pos-0.60-vel-0.40__detail.png]]
##### Summary comparison between position and velocity feedback perturbations
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/force-net-max-during-pert__pert-amp-pos-1.80-vel-1.20.png]]
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/force-net-max-after-pert__pert-amp-pos-1.80-vel-1.20.png]]

#### Trained on  random constant fields

##### Max control force during the perturbation

Consider the max net control force applied by the network during an impulse perturbation to either position or velocity feedback:

![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/force-net-max-during-pert__pos-pert.png]]
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/force-net-max-during-pert__vel-pert.png]]

The increase in gain with perturbation training is stronger for perturbations to velocity feedback, which makes sense – velocity has an integral influence on position. 

##### Max control force after the perturbation
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/force-net-max-after-pert__pos-pert.png]]
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/force-net-max-after-pert__vel-pert.png]]

- Now the effect due to the position feedback perturbation is larger. 
- I assume this is probably because the network has accelerated the point mass in the opposite direction to the perturbation, and after the position perturbation is released, it also has an effective velocity perturbation (caused by itself) to respond to. 
- So some of the gain here may actually be velocity feedback gain. 
- This could be demonstrated by comparing the expected control gains for the velocities at the end of the perturbation period with those due to direct velocity perturbations.
- Consider the velocity response to a 1.8 position feedback impulse, and note that the negative velocity at the end of the perturbation (grey region) is absolutely larger for stronger training field stds. 

![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/v-parallel__pert-amp-1.7999999523162842.png]]
##### Comparison of force profiles
###### For a velocity feedback perturbation

![[F-parallel__pert-amp-1.2000000476837158 1.png]]

![[F-orthogonal__pert-amp-1.2000000476837158 1.png]]
Interesting that the variation in control force at steady state is higher for random-trained than curl-trained networks.

##### Comparison of force profiles during perturbation for position versus velocity feedback impulses

For 1.8 and 0.6.
![[F-parallel__pert-amp-pos-1.80-vel-1.20__detail 1.png]]
![[F-parallel__pert-amp-pos-0.60-vel-0.40__detail 1.png]]

The effect is clearly different than for curl fields, however the increase in velocity feedback control gain with perturbation training is still larger than for position feedback.
###### Comparison of velocity profiles

![[v-parallel__pert-amp-pos-1.80-vel-1.20__detail 1.png]]
![[v-parallel__pert-amp-pos-0.60-vel-0.40__detail 1.png]]

##### Summary comparison between position and velocity feedback perturbations

- Compared to curl-trained networks, the control gains on velocity perturbations also increase, but not to the same degree. 
- Here they end up being equal for position versus velocity feedback perturbation, but that is not that significant since it depends on the choice of impulse amplitude.`

![[force-net-max-during-pert__pert-amp-pos-1.80-vel-1.20 1.png]]
![[force-net-max-after-pert__pert-amp-pos-1.80-vel-1.20 1.png]]
##### Summary

The overall conclusion seems to be that training on perturbations induces higher control gains, and that this effect is stronger on perturbations to velocity than position feedback.

## Additional potential analyses

### Early vs. late perturbations during reaching

Difference between early vs. late perturbations *during reaching* to see if response is the same at different points during the reach.  

.g. given that the maximal forward force output happens in the first couple of timesteps, and is invariant to the disturbance magnitude, does that mean that a disturbance that is only active at the very beginning has a different influence on the response? 

### Low priority
#### Enforce rotational invariance of the learned strategy

This is motivated by the slightly different responses of the network when a feedback perturbation is in different directions. Is it possible to train the network so that its response in the x direction is just like a rotated version of its response in the y direction?

Some ideas:

- Provide the RNN inputs in a polar representation
- Make the RNN output the force vector in a polar representation, by forcing their trigonometric conversion to x/y 
- Add an explicit term to loss function; e.g. on each batch, evaluate on a big center-out set, and penalize for the difference between the control vectors when they are all rotated to lie in the same reach direction
- Modify the network architecture to enforce symmetry. I’m not sure exactly how to do this.