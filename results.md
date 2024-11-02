---
created: 2024-09-24T10:14
updated: 2024-11-02T09:51
---

This is where I’m outlining the modeling results for this project, in the sequence that they will appear in publication.

## 1

### 1.1: Training networks on different disturbance levels

#### No fields

##### No noise, no delay
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/1-1__loss-history__curl__std-0__replicates-10.png]]
##### 0.04 noise, no delay
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/1-1__loss-history__random__std-0__replicates-10.png]]
Clearly noise affects the balance of the loss terms, in particular it puts a floor on the final velocity error (makes sense since due to the motor noise), and likewise it also increases the position error a bit.
##### 0.1 noise, no delay

More noise → higher floor on the velocity and position errors, but qualitatively it looks like the overall evolution is similar, in particular considering the hidden and control force errors remaining so similar in evolution.
![[1-1__loss-history__curl__std-0__replicates-10 1.png]]
##### Zero noise, 2 step delay

This looks almost identical to the zero noise, zero delay case. Presumably, in the absence of unpredictable disturbances, the learning process is essentially identical.
![[1-1__loss-history__curl__std-0__replicates-10 2.png]]
##### Zero noise, 4 step delay

Again, almost identical to the zero noise, zero delay case.
![[1-1__loss-history__random__std-0__replicates-10 1.png]]
##### 0.1 noise, 4 step delay

TODO

#### Curl fields

##### Zero noise, no delay

Curl field std. 0.8
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/1-1__loss-history__curl__std-0.8__replicates-10.png]]
It looks like the most interesting policy developments might happen between iterations 10 and 100.

Curl field std. 1.6
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/1-1__loss-history__curl__std-1.6__replicates-10.png]]
Curl field std. 2.4. At this level, the fields appear to be too strong for the models to converge, at least on average.
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/1-1__loss-history__curl__std-2.4__replicates-10.png]]
It is clear from the loss distribution over replicates that everything is more or less fine except at the highest curl std:
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/best-loss-distn-by-replicate.png]]
##### 0.04 noise, no delay

Curl std. 0.8. 
![[1-1__loss-history__curl__std-0.8__replicates-10 1.png]]
Curl std. 1.6

![[1-1__loss-history__curl__std-1.6__replicates-10 1.png]]
Curl std. 2.4
![[1-1__loss-history__curl__std-2.4__replicates-10 1.png]]
![[best-loss-distn-by-replicate 1.png]]
##### 0.1 noise, no delay

Curl std 0.8
![[1-1__loss-history__curl__std-0.8__replicates-10 2.png]]
Curl std. 1.6
![[1-1__loss-history__curl__std-1.6__replicates-10 2.png]]
Curl std. 2.4
![[1-1__loss-history__curl__std-2.4__replicates-10 2.png]]
Again, qualitatively this is as expected given the lower-noise conditions.
![[best-loss-distn-by-replicate 2.png]]
##### Zero noise, 2 step delay

Curl std. 0.8. Doesn’t look very different from the zero noise, zero delay case.
![[1-1__loss-history__curl__std-0.8__replicates-10 3.png]]
Curl std. 1.6. Now things are looking different from the zero noise, zero delay case. Clearly some or all of the replicates are not stable across training. However, notice that the state errors initially decrease before increasing again.
![[1-1__loss-history__curl__std-1.6__replicates-10 3.png]]
Curl std. 2.4. Apparent divergence. In particular, the hidden loss appears to be hitting a ceiling, which suggests the tanh units are saturating.
![[1-1__loss-history__curl__std-2.4__replicates-10 3.png]]

Comparing the losses across replicates on the final iteration versus the best iteration (for each replicate), it is easy to see that the std. 2.4 models simply diverge, whereas the std. 1.6 models reach a low-ish loss at some point during training.
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/final-loss-distn-by-replicate.png]]
![[best-loss-distn-by-replicate 3.png]]
##### Zero noise, 4 step delay

Curl std. 0.8. Some minor signs of instability towards the end.
![[1-1__loss-history__curl__std-0.8__replicates-10 4.png]]
Curl std. 1.6. More pronounced and definitive divergence than the equivalent condition in the 2-step delay case.
![[1-1__loss-history__curl__std-1.6__replicates-10 4.png]]
The std. 2.4 case is as expected from the 2-step delay case.

#### Random constant fields

##### Zero noise, zero delay

Std. 0.01. Very similar to the no-fields case.
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/1-1__loss-history__random__std-0.01__replicates-10.png]]
Std. 0.1. Some qualitative changes happening in the early period.
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/1-1__loss-history__random__std-0.1__replicates-10.png]]
Std. 1.0. Even more pronounced changes in the early period. 
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/1-1__loss-history__random__std-1.0__replicates-10.png]]
Even at the highest training std., the total loss is small. Presumably if we increased the field strength enough, then it would start to trade off harder with weight decay in the output layer in order to maintain sufficient steady state controlf force. I don’t think there’s a hard ceiling on the control forces though?![[best-loss-distn-by-replicate 4.png]]
##### 0.1 noise, zero delay

Std. 1.0. Mostly just puts a floor on the effector errors, as expected. 
![[1-1__loss-history__random__std-1.0__replicates-10 1.png]]
Systematic increase in the loss with the std, though overall the values are small.
![[best-loss-distn-by-replicate 5.png]]

##### Zero noise, 4 steps delay

Std. 1.0. The delay is not nearly as problematic as it was for the curl field, certainly because the field here is constant and does not interact with the policy in a time-delayed way.
![[1-1__loss-history__random__std-1.0__replicates-10 2.png]]
Std. 2.0. 
![[10 Projects/10 PhD/41 RNNs learn robust policies/results.assets/1-1__loss-history__random__std-2.0__replicates-10.png]]
![[best-loss-distn-by-replicate 6.png]]
### Example center-out sets

### Aligned reach comparisons

###

### 1.2b: Evaluating on feedback perturbations

#### Comparison of profiles for pos vs. vel perturbations, during the perturbation period

This is to show the differential effect of disturbance training on the response to perturbation of different feedback variables.

In particular,

1) training disturbances seem to have little effect on the peri-perturbation response to position feedback perturbations, whereas they have a significant positive effect on response to velocity feedback perturbations;

![[file-20241024161811852.png]]