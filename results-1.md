---
created: 2024-09-24T10:14
updated: 2024-11-03T20:53
---

This is where I’m outlining the modeling results for this project, in the sequence that they will appear in publication.

## 1.1: Training networks on different disturbance levels

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

###### All replicates
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/curl-field-0.0__curl-train-std-0__eval-0.png]]
##### Trained on curl fields
![[curl-amp-0.0__curl-train-std-1.6__rep-3__eval-0.png]]
###### All replicates

Interesting that the variance in the control forces is a bit higher between replicates than it was for the control network. Note that the control forces are larger, but that this is the no-noise condition, so this effect cannot be due to multiplicative noise.
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/curl-field-0.0__curl-train-std-1.6__eval-0.png]]
##### Trained on random constant fields
![[random-amp-0.0__random-train-std-1.0__rep-0__eval-0.png]]
###### All replicates

The variance in the control forces is even greater than for networks trained on curl fields. Again, this has nothing to do with noise.
![[random-field-0.0__random-train-std-1.0__eval-0.png]]
#### Curl field perturbation
##### No training perturbation
![[curl-amp-4.0__curl-train-std-0__rep-6__eval-0.png]]
###### All replicates
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/curl-field-4.0__curl-train-std-0__eval-0.png]]
###### Evaluated with system noise

Whatever policy the network learned, its performance appears not to be significantly affected by system noise. Note that the following plot shows multiple evaluations for a single replicate.
![[curl-field-4.0__curl-std-0__replicate-6.png]]
##### Trained on curl fields
![[curl-amp-4.0__curl-train-std-1.6__rep-3__eval-0.png]]
###### All replicates
![[10 Projects/10 PhD/41 RNNs learn robust policies/results-1.assets/curl-field-4.0__curl-train-std-1.6__eval-0.png]]

###### Evaluated with system noise

Similarly to the control case, noise does not make the policy ineffective.
![[curl-field-4.0__curl-std-1.6__replicate-3.png]]
##### Trained on random constant fields

- This is weird. 
- I suppose it is because the attractors that get strengthened by training on random fields are the ones that output a constant force at the target, but because the network is not accustomed to the curl, it tries to correct the error (similarly but a little better than the control network) until it approaches a ring of constant-force attractors that allow it to orbit the target.
- [ ] Try running this eval for twice as long (200 steps) and see if it keeps orbiting or if it become unstable.
![[curl-amp-4.0__random-train-std-1.0__rep-0__eval-0.png]]
###### All replicates
![[curl-field-4.0__random-train-std-1.0__eval-0.png]]
###### Evaluated with system noise

- Noise perhaps has a slightly worse negative effect than it did in the control and curl-trained conditions.
- Probably because the constant-force orbit attractors are sensitive to changes in position.  
- [ ] Also try this one for 200 steps?
![[curl-field-4.0__random-std-1.0__replicate-0.png]]
#### Random constant field perturbation
##### No training perturbation

- [ ] Perhaps the largest perturbation strength should be higher. 
![[random-amp-0.4__random-train-std-0__rep-6__eval-0.png]]
###### All replicates
![[random-field-0.4__random-train-std-0__eval-0.png]]
###### Evaluated with system noise

##### Trained on random constant fields
![[random-amp-0.4__random-train-std-1.0__rep-0__eval-0.png]]
###### All replicates
![[random-field-0.4__random-train-std-1.0__eval-0.png]]
###### Evaluated with noise
![[random-field-0.4__random-std-1.0__replicate-0.png]]
##### Trained on curl fields

- This fares better than the opposite case, where we trained on random fields and evaluated on curl fields. 
- The network is able to reduce most of the deviation caused by the field, and stop very close to the target.
![[random-amp-0.4__curl-train-std-1.6__rep-3__eval-0.png]]
###### All replicates
![[random-field-0.4__curl-train-std-1.6__eval-0.png]]
###### Evaluated with system noise
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



#### Curl field perturbation
##### No training perturbation

###### 2-step delay
![[curl-field-4.0__curl-train-std-0__eval-0 2.png]]

- The rate and severity of the “curl oscillations” is increased by the delay.

###### 4-step delay
![[curl-field-4.0__curl-train-std-0__eval-0 1.png]]

- The control network is totally unstable in this setting. 
- If we look at the unit activities, they are probably saturating, trying to compensate for movements of the effector away from the intended direction, which it always learns about too late to do anything.

##### Trained on curl fields
###### 2-step delay
![[curl-field-4.0__curl-train-std-1.6__eval-0 1.png]]
###### 4-step delay
![[curl-field-4.0__curl-train-std-1.6__eval-0 2.png]]

- Clearly, delay negatively impacts the ability to deal with curl fields
- Presumably: by the time the network receives feedback, the curl has already bent the effect of the network’s previous control away from the intended direction.
- However, the network has to try to up some of its gains to be robust to the perturbations, and this can lead to 
- The network is still better than the control network
- It is interesting that some of the replicates are “loopy”, but these loops are tighter than the control network’s, probably because the control gains are higher.

##### Trained on random constant fields
###### 2-step delay
![[curl-field-4.0__random-train-std-1.0__eval-0 1.png]]

- This is the condition that had the orbits, prior to the delay.
- The addition of a short delay makes this condition unstable.

###### 4-step delay
![[curl-field-4.0__random-train-std-1.0__eval-0 2.png]]

- Total instability/saturation in this condition.

#### Random constant field perturbation

##### No training perturbation
###### 4-step delay
![[random-field-0.4__random-train-std-0__eval-0 1.png]]

- As expected (see below) the delay does not seriously affect the control network’s policy, which ends up being influenced by the constant field more or less like it would have been without the delay. 
- In other words, this condition is to the undelayed control network perturbed by random field, what the delayed unperturbed control network is to the undelayed unperturbed control network.
- Neither the delay nor the field creates an unstable feedback in the system.
##### Trained on curl fields
###### 2-step delay
![[random-field-0.4__curl-train-std-1.6__eval-0 1.png]]
###### 4-step delay
![[random-field-0.4__curl-train-std-1.6__eval-0 2.png]]

- Performance isn’t great, probably because the network was trained in a condition where it was nearly unstable, so the resulting policy is not great at reaching the goal in any case.
- On the other hand, performance in the 2-step case just above was okay, because the network was still able to perform pretty well in its training condition.

##### Trained on random constant fields
###### 2-step delay
![[random-field-0.4__random-train-std-1.0__eval-0 1.png]]
###### 4-step delay
![[random-field-0.4__random-train-std-1.0__eval-0 2.png]]


- Even with a 4-step delay, the performance of the policy learned for compensating for random constant fields is not degraded.
- Presumably this is because there is not a time-sensitive interaction between network actions and “field actions”, like there was for the curl field, given that the field is constant in this case. 

### 0.1 noise, no delay (TODO)

- [ ] Compile some figures suggesting that training on noise doesn’t significantly alter the resulting policies
- However, it might somewhat decrease performance on noise-free conditions? Since the network was trained with a floor on its loss, and thus may not be adapted to minimize state errors below a certain point?

## Aligned reach comparisons
### No noise, no delay



## 1.2b: Evaluating on feedback perturbations

#### Comparison of profiles for pos vs. vel perturbations, during the perturbation period

This is to show the differential effect of disturbance training on the response to perturbation of different feedback variables.

In particular,

1) training disturbances seem to have little effect on the peri-perturbation response to position feedback perturbations, whereas they have a significant positive effect on response to velocity feedback perturbations;

![[file-20241024161811852.png]]