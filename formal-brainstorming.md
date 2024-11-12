---
created: 2024-11-09T00:06
updated: 2024-11-11T19:51
---
What can we demonstrate formally to support our results?

## Plant dynamics

### With constant force fields
$$
\begin{align}
\mathbf{x}_{t+1}&=\mathbf{A}\mathbf{x}_{t}+\mathbf{B}(\mathbf{u}_{t}\mathbf{w}_{t}+\mathbf{z}_{t}+\mathbf{p}^{(r)}) \\
&=\mathbf{A}\mathbf{x}_{t}+\mathbf{B}(\mathbf{u}_{t}\mathbf{w}_{t}+\mathbf{z}_{t})+\mathbf{B}\mathbf{p}^{(r)}
\end{align}

$$
where:

- $\mathbf{A}$ and $\mathbf{B}$ give the point mass dynamics, which are constant within and between trials
- $\mathbf{w}_{t}$ and $\mathbf{z}_{t}$ are respectively the signal-dependent and signal-independent components of the motor noise
- $\mathbf{p}^{(r)}$ is a force which is constant within each trial $r$, but varies between trials 

During training, we sample the vector $\mathbf{p}$ as uniform in direction and Gaussian in Euclidean norm; that is
$$
\mathbf{p}^{(r)}=l^{(r)}\begin{bmatrix}
\cos(\theta^{(r)}) \\ \sin(\theta^{(r)})
\end{bmatrix}
$$
where $l\sim \mathcal{N}(0,\sigma^{2}_{\mathrm{const}})$ and $\theta\sim U(0,2\pi)$ i.i.d. over trials.

#### Multiplicative motor noise

Note that in general the addition of signal-dependent motor noise to the control force is:
$$
\mathbf{u}_{t}+w(\mathbf{u}_t) 
$$
Assume multiplicative motor noise, such that:
$$
\mathbf{u}_{t}+w(\mathbf{u}_t) =\mathbf{u}_{t}+\mathbf{w}_{t}\mathbf{u}_{t}=\mathbf{u}_{t}(1+\mathbf{w}_t)=\mathbf{u}_{t}\mathbf{w}^{\dagger}_{t}
$$
And we simply refer to $\mathbf{w}_{t}^{\dagger}$ as $\mathbf{w}_{t}$.
### Curl force fields

$$
\begin{align}
\mathbf{x}_{t+1}&=\mathbf{A}\mathbf{x}_{t}+\mathbf{B}(\mathbf{u}_{t}\mathbf{w}_{t}+\mathbf{z}_{t}+\mathbf{P}^{(r)}\mathbf{x}_{t})
 \\ 
&=(\mathbf{A}+\mathbf{BP}^{(r)})\mathbf{x}_{t}+\mathbf{B}(\mathbf{u}_{t}\mathbf{w}_{t}+\mathbf{z}_{t}) \\
&=\mathbf{A}\mathbf{x}_{t}+\mathbf{B}(\mathbf{u}_{t}\mathbf{w}_{t}+\mathbf{z}_{t})+\mathbf{B}\mathbf{P}^{(r)}\mathbf{x}_{t}
\end{align}


$$
where $\mathbf{P}^{(r)}$ is the curl force field, which is constant within each trial $r$.

The force exerted by a curl field is
$$
\mathbf{u}_{\mathrm{curl}}=\begin{bmatrix}
0 & c_{yx}  \\
-c_{xy} & 0
\end{bmatrix}\begin{bmatrix}
v_{x} \\ v_{y}
\end{bmatrix}
$$
where $\mathbf{v}$ is velocity, and $\mathrm{sgn}(c_{yx})=\mathrm{sgn}(c_{xy})$ are real factors that scale the curl. We are only concerned with the isotropic case, $c_{yx}=c_{xy}$, such that
$$
\mathbf{u}_{\mathrm{curl}}=c\begin{bmatrix}
0 & 1  \\
-1 & 0
\end{bmatrix}\begin{bmatrix}
v_{x} \\ v_{y}
\end{bmatrix}
$$
And thus we can write:
$$
\mathbf{x}_{t+1}=\mathbf{A}\mathbf{x}_{t}+\mathbf{B}(\mathbf{u}_{t}\mathbf{w}_{t}+\mathbf{z}_{t})+c^{(r)}\mathbf{B}\mathbf{P}\mathbf{x}_{t}
$$
where $\mathbf{P}$ is a constant matrix which is zero everywhere except for the skew-symmetric block needed to transform the velocities to curl forces. During training, $c\sim \mathcal{N}(0,\sigma_{\mathrm{curl}}^{2})$.
## Cost function

$$
J()=
$$

## Other questions

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

