### Mass

`m_sim = 1`
`m_phys = 1 kg`
`scale_M = 1 kg/[sim mass]`

### Time

`dt_sim = 0.01`
`dt_phys = 0.01 s`
`scale_T = 1 s/[sim time]`

### Length 

Based on the length of a reach – assume a physical reach of 20 cm:

`scale_L = 0.2 m / 0.5 [sim length] = 0.4 m/[sim length]`

### Drag coefficient

The drag coefficient has units $[M\cdot T^{-1}]$, or `kg/s` in SI units.

A typical value is `10 kg/s`; thus the simulation value should be `(10 kg/s) * scale_T / scale_M`, or `10 [sim mass / sim time]`.


### First-order filter time constant

Typical value: `50 ms`

Times are 1:1 so this is `0.05` in sim units. 

### Noise stds.



### Force fields


