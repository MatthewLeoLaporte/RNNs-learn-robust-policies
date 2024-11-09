---
created: 2024-11-09T00:06
updated: 2024-11-09T00:38
---
What can we demonstrate formally to support our results?

## Relatively easy questions 

- How does the level of system noise bound the loss?

## Things I don’t understand

What form/direction would an “asymptotic result” take? 

- Do any nearby examples involving RNNs come to mind?
- Ideally on the more practical side

What would we be deriving bounds for? 

- The loss? 
- Arbitrary functions of state variables? 

And what would we be assuming, in deriving those bounds?

### What simplifications do we need to make?

In particular, I imagine that it will be easiest to treat things linearly.

But my actual networks are non-linear (GRU or tanh vanilla). 

- How should I think, about how much linear results will apply to them?
- Or how best to approach deriving results with nonlinearities? 
- Local linearization?

### Can we treat the specific form of the disturbances? 

1. Both random constant fields and random curl fields have some similar effects, and this is presumably because they both induce model uncertainty
2. ~~On the other hand, system noise does not induce model uncertainty – more like measurement uncertainty~~ 
3. Note that while balanced curl fields are in a sense symmetric, system noise is *more* locally symmetric
4. A more robust controller should be more sensitive to feedback perturbations, but hopefully be relatively insensitive to high-frequency perturbations
5. Curl fields have some effects that constant fields do not – such as the possibility of oscillations due to feedback
6. How should the oscillations due to curl fields vary in frequency? presumably this is a function of the curl field strength, the feedback gains and maximal control forces, as well as any delays

> [!note]
> I crossed out #2 because I think the right way to see this is in terms of frequencies: there are model-free strategies to stabilize a system to unpredicted low-frequency deviations. 
> 
> This may also help to explain why robust networks have higher control gains on velocity than position feedback.

Can we formalize how different types/frequencies of disturbance should affect loss gradients? i.e. why locally-symmetric noise has much less effect on the policy than batch-symmetric but trial-biased force fields?

**Perhaps if we consider a local linearization of the network, and given that the point mass and the force fields are also linear, we then have a closed-loop system whose transfer function could be analyzed.** 

## How to proceed

- I would like to derive some results myself
- However I am not under illusions about my ability to generate mathematical proofs, which is pretty weak
- And I am not very confident in how much I will actually be able to formalize in any case
- However I am willing to try.

If we could have meetings every week or so for a the next 2-3 weeks, that would give me some motivation to at least try to get somewhere with this – knowing that someone much more competent than me in this area, may be able to give me a bit of advice along the way.

- But I also understand that you are probably busy with your own interests
- Of course, how much you participate is up to you