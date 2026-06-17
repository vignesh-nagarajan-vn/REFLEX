# EDL Simulator v1 HTML File
- 5-part EDL (Endogenous Distribution Learning) framework
- Linear-quadratic OTC bond model with analytical fixed point at θ* = (2.6 − 0.5α)/(1 − α)
- Convergence theorem: stable iff α < α_c = 1, geometric rate α^t
- Sample image of execution below:

<img width="971" height="1153" alt="image" src="https://github.com/user-attachments/assets/d1a3a0e2-cd55-45dd-854e-badcd1769deb" />

A market simulation model where trader strategies (parameterized by adversarialness α) determine stability: below α=1 the system converges to equilibrium, above it diverges into chaos. 

Currently validated at α=0.45. Next steps are adding dealer policy mechanics and formal stability theorems to better predict the direction of market chaos.

 The novelty lies in using a single adversarialness parameter as an inflection-based predictor of market regime, bridging behavioral trader modeling with formal dynamical systems theory.
