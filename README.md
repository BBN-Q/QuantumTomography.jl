[![Build Status](https://travis-ci.org/BBN-Q/QuantumTomography.jl.svg?branch=master)](https://travis-ci.org/BBN-Q/QuantumTomography.jl)

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
Choi matrix of a completely positive, trace preserving map, and
measurements are constrained to be given by POVM elements.

## API

Each different tomography method is associated with a type. Objects of
this type must be instantiated before reconstruction can be performed
with a call to `fit()`. These objects are also needed to make
predictions about tomography experiments with `predict()`. Currently available
tomography methods are

+ `FreeLSStateTomo`: Unconstrained least-squares state tomography.

+ `LSStateTomo`: Least-squares state tomography constrained to yield physical states.

+ `MLStateTomo`: Maximum-likelihood state tomography (including the option for entropy maximization or hedging).

## Examples

### Constrained least-squares tomography

In order to perform quantum state tomography, we need an
informationally complete set of measurement effects. In the case of a single
qubit, that can be given by the eigenstates of the 3 Pauli operators.
```julia
julia> using Cliffords, QuantumTomography

julia> obs = Matrix[ (complex(Pauli(i))+eye(2))/2 for i in 1:3 ];

julia> append!(obs, Matrix[ (-complex(Pauli(i))+eye(2))/2 for i in 1:3 ]);

julia> tomo = LSStateTomo(obs);
```
We choose some random pure state to generate the ficticious experiment
```julia
julia> using RandomQuantum, QuantumInfo

julia> ψ  = rand(FubiniStudyPureState(2));

julia> normalize!(ψ)
2-element Array{Complex{Float64},1}:
 0.264298+0.850605im
 0.449897-0.0648884im

julia> ρ = projector(ψ)
2x2 Array{Complex{Float64},2}:
  0.793382+0.0im       0.0637123+0.399834im
 0.0637123-0.399834im   0.206618+0.0im     
```
Predict the expectation values of the observations for some hypothesized ρ
```julia
ideal_means = predict(tomo, ρ) |> real
```
With these in hand, we can finally reconstruct `ρ` from the observed expectation values and variances.
```julia
julia> fit(tomo, ideal_means, ones(6))
(
2x2 Array{Complex{Float64},2}:
 0.793382-9.39512e-25im                0.0637123+0.399834im
         0.0637123-0.399834im  0.206618+7.98287e-25im      ,

3.730685819507896e-11,:Optimal)
```

### Constrained maximum-likelihood tomography (with maximum entropy and hedging options)

Using the data generated above, we can instead choose to reconstruct the state
by maximizing the likelihood function for some set of binomial observations.
```julia
julia> using Distributions
julia> ml_tomo = MLStateTomo(obs)
julia> freqs = Float64[rand(Binomial(10_000, μ))/10_000 for μ in ideal_means[1:3]]
julia> append!(freqs,1-freqs)

julia> fit(ml_tomo, freqs)
(
2x2 Array{Complex{Float64},2}:
 0.789799-7.92259e-17im                0.0641298+0.401799im
         0.0641298-0.401799im  0.210201-4.28412e-17im      ,

-1.5215022154657007,:Optimal)
```
If the observations are incomplete (in the sense that they do not uniquely specify
the quantum state), one can still perform reconstruction by maximizing a mixture
of the likelihood and the entropy of the resulting state (see PRL 107 020404 2011).
In this package, this would correspond to
```julia
julia> fit(ml_tomo, freqs, λ=1e-3)
(
2x2 Array{Complex{Float64},2}:
 0.789155-2.68147e-17im                0.0639152+0.401322im
         0.0639152-0.401322im  0.210845-9.18039e-18im      ,

-1.5215005466837999,:Optimal)
```

Constrained maximum-likelihood also suffers from biasing towards low
rank states.  This can be avoided by *hedging* (see Blume-Kohout, PRL
105, 200504 2010), which essentially follows a modification of
Laplace's rule to penalize low rank estimates. Hedging can be enabled
by using the experimental fitting routine `fitA` with `MLStateTomo`:
```julia
julia> QuantumTomography.fitA(ml_tomo, freqs)
(
2x2 Array{Complex{Float64},2}:
 0.789155-2.68147e-17im                0.0639152+0.401322im
         0.0639152-0.401322im  0.210845-9.18039e-18im      ,

-1.5215005466837999,:Optimal)
```
## TODO

- [ ] Implement least-squares and ML process tomography
- [ ] Implement compressed sensing state and process tomography

## Copyright

Raytheon BBN Technologies.

## License

Apache Lincense 2.0 ([summary](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)))

## Authors

Marcus Silva (@marcusps on GitHub)
