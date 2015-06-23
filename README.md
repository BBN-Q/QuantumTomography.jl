# QuantumTomography.jl

This is a package that provides basic routines to perform tomography
of finite dimensional systems.

The tomography technique used relies on the assumption that the
ensemble of observations is large enough so that the estimated means
have a Gaussian distribution (to good approximation).  This assumption
is not essential, but it is convenient for the use of [Convex.jl]() as
a solver. 

Given the estimated means and variances of the observations,
tomography proceeds by finding the state, process, or measurements
that minimizes the chi-squared statistic under the constraints for
"physicality" of those objects. Namely, states are constrained to be
given by positive semidefinite matrices with unit trace, processes are
constrained to be given by the Choi matrix of a completely positive,
trace perserving map, and measurements are constrained to be given by
POVM elements.

## API


+ `state_tomo_lsq()`: state tomography without constrait to physical states

+ `state_tomo_ml()`: state tomography constrainted to physical states (positive and unit trace)

+ `proc_tomo_lsq()`: process tomography with constraint to physical processes

+ `proc_tomo_ml()` :

+ `meas_tomo_lsq()`

+ `meas_tomo_ml()`


## Examples

In order to perform quantum state tomography, we need an
informationally complete set of observables. In the case of a single
qubit, that can be given by the 3 Pauli operators.
```julia
obs = Matrix[pauli(1), pauli(2), pauli(3)]
```
We choose some random pure state to generate the ficticious experiment 
```julia    
\psi = randn(2)+1im*randn(2); \psi=\psi/norm(\psi,2)
ρ = psi*psi'
```
We can compute the expectation values of the observables for this state using Born's rule
```julia
ideal_means = Float64[ real(trace(o*\rho)) for o in obs ]
```julia
With these in hand, we can finally reconstruct `\rho` from the observed expectation values and variances.
```
qst_ml(obs, ideal_means, ideal_vars);
```

## TODO

- [ ] Implement proper ML tomography without Gaussian assumption
- [ ] Implement hedged ML tomography

## Copywrite

Raytheon BBN Technologies.

## License

Apache?

## Authors

Marcus Palmer da Silva (@marcusps on GitHub)