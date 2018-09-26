using Test,
      Cliffords,
      Distributions,
      QuantumInfo,
      QuantumTomography,
      RandomQuantum,
      SchattenNorms

import Random

function qst_test_setup()
    obs = [ (complex(Pauli(i))+QuantumInfo.eye(2))/2 for i in 1:3 ]
    append!(obs, [ (-complex(Pauli(i))+QuantumInfo.eye(2))/2 for i in 1:3 ])

    ρ = .98*projector(rand(FubiniStudyPureState(2)))+.02*rand(HilbertSchmidtMixedState(2))

    return ρ, obs
end

function test_qst_freels(ρ, obs; n=10_000, asymptotic=false)
    tomo = FreeLSStateTomo(obs)

    asymptotic_means = real(predict(tomo,ρ))

    samples = asymptotic ? asymptotic_means : Float64[ rand(Distributions.Binomial(n,μ))/n for μ in asymptotic_means ]

    ρest, obj, status = fit(tomo, samples)

    return status, trnorm(ρ-ρest), obj, ρest
end

function test_qst_freels_gls(ρ, obs; n=10_000, asymptotic=false)
    tomo = FreeLSStateTomo(obs)

    asymptotic_means = real(predict(tomo,ρ))

    samples = asymptotic ? asymptotic_means : Float64[ rand(Distributions.Binomial(n,μ))/n for μ in asymptotic_means ]
    sample_mean = samples
    sample_var  = n*(samples - samples.^2)/(n-1)

    ρest, obj, status = fit(tomo, sample_mean, asymptotic ? ones(length(samples)) : sample_var)

    return status, trnorm(ρ-ρest), obj, ρest
end

function test_qst_ls(ρ, obs; n=10_000, asymptotic=false)
    tomo = LSStateTomo(obs)

    asymptotic_means = real(predict(tomo,ρ))

    samples = asymptotic ? asymptotic_means : Float64[ rand(Distributions.Binomial(n,μ))/n for μ in asymptotic_means ]
    sample_mean = samples
    sample_var  = n*(samples - samples.^2)/(n-1)

    ρest, obj, status = fit(tomo, sample_mean, asymptotic ? ones(length(samples)) : sample_var);

    return status, trnorm(ρ-ρest), obj, ρest
end

function test_qst_ml(ρ, obs; n=10_000, β=0.0, asymptotic=false, alt=false, maxiter=5000, ϵ=1000)
    tomo = MLStateTomo(obs,β)

    asymptotic_means = real(predict(tomo,ρ))

    samples = asymptotic ? asymptotic_means[1:3] : Float64[rand(Distributions.Binomial(n,μ))/n for μ in asymptotic_means[1:3]]
    append!(samples, 1 .- samples)

    ρest, obj, status = fit(tomo, samples, maxiter=maxiter, δ=1/ϵ, λ=β)

    return status, trnorm(ρ-ρest), obj, ρest
end

function test_qst_hml(ρ, obs; n=10_000, β=0.0, asymptotic=false, alt=false, maxiter=5000, ϵ=1000)
    tomo = MLStateTomo(obs,β)

    asymptotic_means = real(predict(tomo,ρ))

    samples = asymptotic ? asymptotic_means[1:3] : Float64[rand(Distributions.Binomial(n,μ))/n for μ in asymptotic_means[1:3]]
    append!(samples,1-samples)

    ρest, obj, status = fitB(tomo, samples)

    return status, trnorm(ρ-ρest), obj, ρest
end


function test_qpt_free_lsq(n=1000; E=zeros(ComplexF64,0,0), asymptotic=false)

    prep = map(projector, [ [1,0],
                            [0,1],
                            1/sqrt(2)*[1,1],
                            1/sqrt(2)*[1,-1],
                            1/sqrt(2)*[1,-1im],
                            1/sqrt(2)*[1,1im] ] )

    u = rand(ClosedHaarEnsemble(2))

    obs = map(ψ->projector(u*ψ), [ [1,0],
                                   [0,1],
                                   1/sqrt(2)*[1,1],
                                   1/sqrt(2)*[1,-1im] ] )

    if size(E) == (0,0)
        E = liou(rand(RandomQuantum.ClosedHaarEnsemble(2)))
    end

    tomo = FreeLSProcessTomo(prep, obs)

    asymptotic_means = predict(tomo, E)

    means = asymptotic ? asymptotic_means : map(p->rand(Distributions.Binomial(n,p))/n, asymptotic_means)
    vars = means .* (1 .- means) / n

    Eest, obj, status = asymptotic ? fit(tomo, asymptotic_means) : fit(tomo, means, vars)

    choi_err = liou2choi(Eest - E)

    #println("Status                  : $(status)")
    #println("Diamond norm lower bound: $(snorm(choi_err,1))")
    #println("χ² error                : $(obj)")
    #println("Eigvals ρ:")
    #for ev in eigvals(E)
    #    println(ev)
    #end
    #println("Eigvals ρest:")
    #for ev in eigvals(Eest)
    #    println(real(ev))
    #end

    return status, snorm(choi_err,1), obj, Eest
end

function test_qpt_lsq(n=1000; E=zeros(ComplexF64,0,0), asymptotic=false)

    prep = map(projector, [ [1,0],
                            [0,1],
                            1/sqrt(2)*[1,1],
                            1/sqrt(2)*[1,-1],
                            1/sqrt(2)*[1,-1im],
                            1/sqrt(2)*[1,1im] ] )

    u = rand(ClosedHaarEnsemble(2))

    obs = map(ψ->projector(u*ψ), [ [1,0],
                                   [0,1],
                                   1/sqrt(2)*[1,1],
                                   1/sqrt(2)*[1,-1im] ] )

    if size(E) == (0,0)
        E = liou(rand(RandomQuantum.ClosedHaarEnsemble(2)))
    end

    tomo = LSProcessTomo(prep, obs)

    asymptotic_means = predict(tomo, E)

    means = asymptotic ? asymptotic_means : map(p->rand(Distributions.Binomial(n,p))/n, asymptotic_means)
    vars = means .* (1 .- means) / n

    Eest, obj, status = asymptotic ? fit(tomo, asymptotic_means) : fit(tomo, means, vars)

    choi_err = liou2choi(Eest - E)

    #println("Status                  : $(status)")
    #println("Diamond norm lower bound: $(snorm(choi_err,1))")
    #println("χ² error                : $(obj)")
    #println("Eigvals ρ:")
    #for ev in eigvals(E)
    #    println(ev)
    #end
    #println("Eigvals ρest:")
    #for ev in eigvals(Eest)
    #    println(real(ev))
    #end

    return status, snorm(choi_err,1), obj, Eest
end

function test_trb(da,db)
    r = rand_mixed_state(da*db)
    LinearAlgebra.norm(LinearAlgebra.tr(r,[da,db],2)-mat(trb_sop(da,db)*vec(r)),1)
end

ρ, obs = qst_test_setup()

kmax = 100
result = zeros(kmax,6)

Random.seed!(314159)

@testset "Set 1" begin

for k = 1:kmax

    ρ = 0.98 * projector(rand(FubiniStudyPureState(2))) + 0.02 * rand(HilbertSchmidtMixedState(2))

    status, enorm, obj, ρest = test_qst_freels(ρ, obs, asymptotic=true)
    result[k,1] = enorm
    # println("Constrained LS with ∞ counts:")
    # println("   status    : $status")
    # println("   error     : $enorm")
    # println("   true state: $ρ")
    # println("   estimate  : $ρest")
    @test status == :Optimal
    @test enorm < 1e-6

    status, enorm, _, ρest = test_qst_freels(ρ, obs, asymptotic=false)
    result[k,2] = enorm
    # println("Constrained LS with 10_000 counts:")
    # println("   status    : $status")
    # println("   error     : $enorm")
    # println("   true state: $ρ")
    # println("   estimate  : $ρest")
    @test status == :Optimal
    @test enorm < 5e-2

    status, enorm, obj, ρest = test_qst_ls(ρ, obs, asymptotic=true)
    result[k,1] = enorm
    # println("Constrained LS with ∞ counts:")
    # println("   status    : $status")
    # println("   error     : $enorm")
    # println("   true state: $ρ")
    # println("   estimate  : $ρest")
    @test status == :Optimal
    @test enorm < 1e-6

    status, enorm, _, ρest = test_qst_ls(ρ, obs, asymptotic=false)
    result[k,2] = enorm
    # println("Constrained LS with 10_000 counts:")
    # println("   status    : $status")
    # println("   error     : $enorm")
    # println("   true state: $ρ")
    # println("   estimate  : $ρest")
    @test status == :Optimal || ( status == :UnknownError && enorm < 5e-2 )
    @test enorm < 5e-2

    status, enorm, _, ρest = test_qst_ml(ρ, obs, asymptotic=true)
    result[k,3] = enorm
    # println("Strict ML with mean counts:")
    # println("   status    : $status")
    # println("   error     : $enorm")
    # println("   true state: $ρ")
    # println("   estimate  : $ρest")
    @test status == :Optimal
    @test enorm < 5e-2

    status, enorm, _, ρest = test_qst_ml(ρ, obs, asymptotic=false)
    result[k,4] = enorm
    # println("Strict ML with 10_000 counts:")
    # println("   status    : $status")
    # println("   error     : $enorm")
    # println("   true state: $ρ")
    # println("   estimate  : $ρest")
    @test status == :Optimal || status == :MaxIter
    @test enorm < 5e-2

    status, enorm, _, ρest = test_qst_ml(ρ, obs, β=0.001, asymptotic=true)
    result[k,5] = enorm
    # println("Hedged ML with mean counts and β=0.04:")
    # println("   status    : $status")
    # println("   error     : $enorm")
    # println("   true state: $ρ")
    # println("   estimate  : $ρest")
    @test status == :Optimal
    @test enorm < 5e-2

    status, enorm, _, ρest = test_qst_ml(ρ, obs, β=0.001, asymptotic=false)
    result[k,6] = enorm
    # println("Hedged ML with 10_000 counts and β=0.04:")
    # println("   status    : $status")
    # println("   error     : $enorm")
    # println("   true state: $ρ")
    # println("   estimate  : $ρest")
    @test status == :Optimal
    @test enorm < 5e-2

    # status, enorm, _, ρest = test_qst_hml(ρ, obs, β=0.001, asymptotic=false)
    # result[k,6] = enorm
    # println("Hedged ML with 10_000 counts and β=0.04:")
    # println("   status    : $status")
    # println("   error     : $enorm")
    # println("   true state: $ρ")
    # println("   estimate  : $ρest")
    # println(enorm)
    #@test status == :Optimal
    #@test enorm < 5e-2
end

end

@testset "Set 2" begin

for k = 1:kmax

    # E = liou(rand(ClosedHaarEnsemble(2)))
    E = rand(OpenHaarEnsemble(2,3))

    status, enorm, _, Eest = test_qpt_free_lsq(10_000, E=E, asymptotic=true)
    @test status == :Optimal
    @test enorm < 1e-3

    status, enorm, _, Eest = test_qpt_free_lsq(100_000, E=E, asymptotic=false)
    @test status == :Optimal
    @test enorm < 1.1e-2

    status, enorm, _, Eest = test_qpt_lsq(10_000, E=E, asymptotic=true)
    @test status == :Optimal
    @test enorm < 1e-3

    status, enorm, _, Eest = test_qpt_lsq(100_000, E=E, asymptotic=false)
    @test_broken status == :Optimal || ( status == :UnknownError && enorm < 1.1e-2 )
    @test_broken enorm < 1.1e-2

    #println(result[k,:])
end
#@printf "\n"

end

#println(result)
