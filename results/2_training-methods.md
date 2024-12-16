
There are some differences between the three training methods I’ve [[methods#Training methods|tried]]. 

Especially in the case of constant fields, there are some differences between these that are not super clear to me yet, which is why I want to outline them here.

## Constant fields

### General observations

- There is a training std. which induces a good spread of robustness behaviour across context inputs. Mostly this seems to lie between 0.03 and 0.08.
- For larger training stds., depending on the training method, even negative context inputs may produce solutions that look very similar to baseline in terms of the maximal and summed velocities and forces, and mainly differ in terms of geometry and steady-state outputs.
- “Hyper-robustness” does not appear as consistently as with curl fields; for BCS and PAI there are some overshoots, but they are not consistent with PAI. However, even without overshoot the steady-state attractors may be strengthened. This remains to be seen in the network analysis.

### Binary context switch (BCS)

> [!Info]
> Here, 25% of trials are perturbed. The perturbation is drawn from a zero-mean, `field std.`-std normal distribution. If the field is perturbed, the network gets a context input of 1, otherwise 0.

> [!NOTE]+
> These were trained and evaluated with 0.01 noise std.
> 
> Also, `p_perturbed` is 0.25 for both BCS and DAI methods, and 1 for DAI, whereas elsewhere I have used 0.25 only for BCS and 1 for DAI and PAI. In the end we should settle on a consistent value; however note that values less than 1 do not work well for curl fields, for whatever reason. 

#### Aligned trajectories

Here, we do see a spread of robustness behaviour across context input. Here is std. 0.32 (which is very similar to std. 0.16):

![[file-20241130102311669.png]]

Note that we also see “hyperrobust” behaviour, which in this case looks like oscillation of the control force around its fixed point, and greater endpoint error in the opposite direction to the force field.

The spread across context inputs is clear significantly earlier, around std. 0.02.
![[file-20241130102405458.png]]

At std 0.01, it is almost negligible:
![[file-20241130102548219.png]]

Compare that to the control std 0.0:

![[file-20241130104144094.png]]

Importantly, for this training method, the context-0 behaviour is similar across training conditions. This is something we’d like to see because it implies that the perturbations are not differentially influencing the baseline strategy.

![[file-20241130102658559.png]]

We also see “anti-robustness” at negative context inputs. Notice that in an std-dependent way, the controls get weaker and the trajectories lean further away from the goal:

![[file-20241130102822910.png]]

The context-1 spread makes sense in the position profiles, becoming more robust as the std increases. However, the force and velocity profiles are a bit more complicated. Note that the maximal control force first increases to reach a maximum at 0.03 and 0.04, then decreases again. This appears to coincide with the position endpoint making contact with the target, and probably has to do with a shift in network dynamics to a “sufficiently elliptical” output profile.
![[file-20241130104300018.png]]

The pattern is similar at context 2, with std-dependent hyperrobust overcorrections. Note however that none of the trajectories are obviously “better” at context 2; they do curve toward the goal sooner, but are also more unstable.
![[file-20241130104333171.png]]
#### Measure distributions


> [!question] 
> Is the broadening of the distributions seen in human data? (Or theory?)

Note that the variance in max forward velocity increases with train std.

![[file-20241130104450176.png]]
Here the under-robustness and hyper-robustness are evident. Note the weird bimodality in the higher stds, for the context 2 condition, which is because some trajectories end up hitting negative deviations larger than their largest positive deviation.
![[file-20241130104749781.png]]

Perhaps a clearer demonstration is the *sum* of signed lateral deviations:
![[file-20241130105040337.png]]

The position error is sensible. Note that with under-robustness and hyper-robustness, the error and its variance go up. Also note that for lower train stds., we are still continuing to get more robust at context 2.
![[file-20241130105214643.png]]


> [!attention]+
> One strange thing in the above plot is the spread seen in the mean position error at context 0. Note that this is similar to the relationship we saw in the control forces at context 1: first they increase up to std 0.03-0.04, then they decrease again. 

The max net control force follows the relationship we’ve already discussed; it increases until std 0.03-0.04, then decreases a bit and becomes higher variance:

![[file-20241130105738460.png]]

The sum of net control forces varies by about 5-10% across training conditions and context inputs (note the y axis is not bounded at 0)
![[file-20241130105946217.png]]

### Direct amplitude information (DAI)

#### Aligned trajectories

As expected, the control std 0.0 looks the same as with other methods:
![[file-20241130110904225.png]]And at std 0.01, the spread is only barely present:
![[file-20241130110927407.png]]

But becomes apparent at 0.02:
![[file-20241130110943663.png]]
Unlike the BCS method, at std 0.04 the spread is still continuing to develop:
![[file-20241130111055763.png]]

But by 0.08, everything is tight:
![[file-20241130111135914.png]]

Interestingly, nothing much happens as we continue to increase the std past this point, except that the spread in the negative contexts comes back just a bit. Here is std 0.32:
![[file-20241130111230084.png]]

Looking at the trajectories compared by train std., it is clear in the context -2 comparison that there is some kind of switch between 0.04 and 0.08
![[file-20241130111449138.png]]

> [!attention]+
> I think this is unfavourable compared to the BCS method, since it means that the training condition (std) is significantly influencing what is happening when the context input is 0.
> 
> This may not be an issue for further analyses in which we fix the train std., and are only concerned with variation in the context input. In that case, we may be more interested in there being a good spread of behaviour across context inputs. However, we don’t really see much under- or hyper-robustness using this training method either. (I think this is similar to the “amplitude” method for curl fields – the network has no incentive to extrapolate because it always knows exactly how strong the field is going to be.)


At context 0, instead of seeing that all the training conditions are similar, we see a developing spread and a saturation above about 0.04:
![[file-20241130111538432.png]]

At context 1 and context 2, we have no hyperrobustness at all, just a fully developed spread. Here is context 2, though context 1 looks similar:
![[file-20241130111649114.png]]
#### Measure distributions

The first thing to note is that in what follows, compared to the BCS method, there is much less of an effect of large train stds on measure variances at extreme context inputs.

The max forward velocity increases (decreases) to 0.04, then reverts back to the control level:
![[file-20241130112357919.png]]

Which tracks with the max net control force:
![[file-20241130113102777.png]]

And **in particular, the max forward velocity looks very similar to the sum of net control forces**:
![[file-20241130113203775.png]]

There is a good spread of largest lateral distance up to 0.04, and above that, everything is relatively robust even at context -2:
![[file-20241130112645439.png]]
Note in particular that compared to BCS, we never go negative. 

The sum of signed deviations is similar, if a bit clearer. This is a nice summary of the shift that happens between 0.04 and 0.08, and how at large context inputs it converges on a spread but no hyper-robustness.
![[file-20241130112858651.png]]

Same with mean position error:
![[file-20241130113011965.png]]

### Probabilistic amplitude information (PAI)

> [!Caution]+
> The result for `p_perturbed=0.25` is strange, so we review it first, and then proceed to the bulk of the results for `p_perturbed=1.0`.

#### `p_perturbed=0.25`

> [!attention] 
> Verify that the context input is actually zero on unperturbed trials

Weirdly, the expected trend is totally reversed in this case. At std 0.08 there is still no spread developed,
![[file-20241130113736813.png]]

But by 0.16 it is clear that the spread is in the “wrong” direction, with negative contexts being the most robust:
![[file-20241130113808577.png]]
Until at 0.32:
![[file-20241130113840641.png]]

Comparing the train stds, consider context -2:
![[file-20241130113913867.png]]
and compare this with context 2:
![[file-20241130113951093.png]]

The measure distributions reflect these strange reversed relationships:
![[file-20241130114042773.png]]
![[file-20241130114105772.png]]
#### Aligned trajectories

Control is as expected
![[file-20241130140346124.png]]

Spread becomes apparent at std 0.02
![[file-20241130140448871.png]]

By 0.04, context 2 has overshot the goal
![[file-20241130140532658.png]]

However at higher stds 0.16 (below) and 0.32, things normalize again, and start to look more like the DAI policies. I assume this is because even with the probabilistic information, there is less uncertainty that the perturbation will be a large one even if the context input is relatively small.
![[file-20241130140617353.png]]

Looking at the train std comparisons, it is clear that at lower train stds that under-robustness occurs for context -2, but at higher stds it does not (relative to baseline, though of course it is still less robust relative to context 0, context 1, etc.)
![[file-20241130140900326.png]]

At context 0 we have a spread, rather than uniformity. Note that 0.08, 0.16, 0.32 all reach the target and have approximately the same profiles. (At context 1, not shown here, the spread moves slightly closer to the goal, with std 0.04 close to overlapping with std ≥0.08)
![[file-20241130140951119.png]]

At context 2, the spread remains and curiously std 0.32 reaches the goal, but 0.08 and 0.16 overshoot.
![[file-20241130141241405.png]]

#### Measure distributions

Max velocity is in line with DAI; up to std ~0.04 the trend develops, followed by regression to baseline.

![[file-20241130141309515.png]]
Similarly with control force:
![[file-20241130141714891.png]]
![[file-20241130141737419.png]]

Lateral distance is in some ways intermediate between BCS and DAI, with only a few negative values appearing for context 2. 
![[file-20241130141432323.png]]
![[file-20241130141558478.png]]
![[file-20241130141617043.png]]