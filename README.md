# QuantumTomography.jl

This is a package that provides basic routines to perform quantum
tomography of finite dimensional systems.

Given the estimated means and variances of the observations,
tomography proceeds by finding the state, process, or measurements
that optimize some figure of merit -- either the likelihood or the χ²
statistic -- often with the additional constraint the reconstruction
be physical (e.g., correspond to a valid quantum mechanical object).

Namely, states are constrained to be given by positive semidefinite
matrices with unit trace, processes are constrained to be given by the
Choi matrix of a completely positive, trace perserving map, and
measurements are constrained to be given by POVM elements.

## API

Each different tomography method is associated with a type. Objects of
this type must be instantiated before reconstruction can be performed
with a call to `fit()`. These objects are also needed to make
predictions about tomography experiments with `predict()`. Currently available
tomography methods are

+ `FreeLSStateTomo`: Unconstrained least-squares state tomography.

+ `LSStateTomo`: Least-squares state tomography constrained to yield physical states.

+ `MLStateTomo`: Maximum-likelihood state tomography (including the option for entropy maximization as well). 

## Examples

### Constrained least-squares tomography

In order to perform quantum state tomography, we need an
informationally complete set of measurement effects. In the case of a single
qubit, that can be given by the eigenstates of the 3 Pauli operators.
```julia
using Cliffords
obs = Matrix[eye(2)+complex(Pauli(1)), 
             eye(2)+complex(Pauli(2)), 
             eye(2)+complex(Pauli(3)), 
	     eye(2)-complex(Pauli(1)),  
	     eye(2)-complex(Pauli(2)),  
	     eye(2)-complex(Pauli(3))]/2
tomo = LSStateTomo(obs)
```
We choose some random pure state to generate the ficticious experiment 
```julia
using RandomQuantum, QuantumInfo
ψ  = rand(FubiniStudyPureState(2)); 
normalize!(ψ)
ρ = projector(ψ)
```
Predict the expectation values of the observations for some hypothesized ρ
```julia
ideal_means = predict(tomo, ρ)
```
With these in hand, we can finally reconstruct `ρ` from the observed expectation values and variances.
```
fit(tomo, ideal_means + σ.*randn(3), σ.^2)
```

### Constrained maximum-likelihood and maximum-entropy tomography

Using the data generated above, we can instead choose to reconstruct the state
by maximizing the likelihood function for some set of binomial observations.
```
ml_tomo = MLStateTomo(obs)
fit(tomo, Float64[Binomial(10_000, μ)/10_000 for μ in ideal_means])
```
If the observations are incomplete (in the sense that they do not uniquely specify
the quantum state), one can still perform reconstruction by maximizing a mixture 
of the likelihood and the entropy of the resulting state (see PRL 107 020404 2011).
In this package, this would correspond to 
```
fit(tomo, Float64[Binomial(10_000, μ)/10_000 for μ in ideal_means], λ=1e-3)
```
## TODO

- [ ] Implement least-squares and ML process tomography
- [ ] Implement compressed sensing state and process tomography

## Copywrite

Raytheon BBN Technologies.

## License

Apache Lincense 2.0 ([summary](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)))

## Authors

Marcus Palmer da Silva (@marcusps on GitHub)
