using Base.Test,
      Cliffords, 
      Distributions,
      QuantumInfo, 
      QuantumTomography, 
      RandomQuantum,
      SchattenNorms,
      SCS

function qst_test_setup()
    obs = Matrix[ (complex(Pauli(i))+eye(2))/2 for i in 1:3 ]
    append!(obs, Matrix[ (-complex(Pauli(i))+eye(2))/2 for i in 1:3 ])

    ρ = .98*projector(rand(FubiniStudyPureState(2)))+.02*rand(HilbertSchmidtMixedState(2))

    return ρ, obs
end

function test_qst_freels(ρ, obs; n=10_000, asymptotic=false)
    tomo = FreeLSStateTomo(obs)
    
    asymptotic_means = real(predict(tomo,ρ))

    samples = asymptotic ? asymptotic_means : [ rand(Binomial(n,μ))/n for μ in asymptotic_means ]
    sample_mean = samples
    sample_var  = n*(samples - samples.^2)/(n-1)
    
    ρest, obj, status = fit(tomo, sample_mean, asymptotic ? ones(length(samples)) : sample_var);
    
    return status, trnorm(ρ-ρest), obj, ρest
end

function test_qst_freels_gls(ρ, obs; n=10_000, asymptotic=false)
    tomo = FreeLSStateTomo(obs)
    
    asymptotic_means = real(predict(tomo,ρ))

    samples = asymptotic ? asymptotic_means : [ rand(Binomial(n,μ))/n for μ in asymptotic_means ]
    sample_mean = samples
    sample_var  = n*(samples - samples.^2)/(n-1)
    
    ρest, obj, status = fit(tomo, sample_mean, asymptotic ? ones(length(samples)) : sample_var, algorithm=:GLS);
    
    return status, trnorm(ρ-ρest), obj, ρest
end

function test_qst_ls(ρ, obs; n=10_000, asymptotic=false)
    tomo = LSStateTomo(obs)

    asymptotic_means = real(predict(tomo,ρ))

    samples = asymptotic ? asymptotic_means : [ rand(Binomial(n,μ))/n for μ in asymptotic_means ]
    sample_mean = samples
    sample_var  = n*(samples - samples.^2)/(n-1)
    
    ρest, obj, status = fit(tomo, sample_mean, asymptotic ? ones(length(samples)) : sample_var);
    
    return status, trnorm(ρ-ρest), obj, ρest
end

function test_qst_ml(ρ, obs; n=10_000, β=0.0, asymptotic=false, alt=false, maxiter=5000, ϵ=1000)
    tomo = MLStateTomo(obs,β)

    asymptotic_means = real(predict(tomo,ρ))
    
    samples = asymptotic ? asymptotic_means[1:3] : Float64[rand(Binomial(n,μ))/n for μ in asymptotic_means[1:3]]
    append!(samples,1-samples)

    ρest, obj, status = fit(tomo, samples, maxiter=maxiter, δ=1/ϵ, λ=β)

    return status, trnorm(ρ-ρest), obj, ρest
end

function test_qst_hml(ρ, obs; n=10_000, β=0.0, asymptotic=false, alt=false, maxiter=5000, ϵ=1000)
    tomo = MLStateTomo(obs,β)

    asymptotic_means = real(predict(tomo,ρ))
    
    samples = asymptotic ? asymptotic_means[1:3] : Float64[rand(Binomial(n,μ))/n for μ in asymptotic_means[1:3]]
    append!(samples,1-samples)

    ρest, obj, status = fitB(tomo, samples)

    return status, trnorm(ρ-ρest), obj, ρest
end


function test_qpt_ml(n=1000;ρ=zeros(Float64,0,0))

    obs  = Matrix[ [0 1; 1 0], [0 -1im; 1im 0], pauli(3) ]
    prep = map(projector,Vector[ [1;0], [0;1], 1/sqrt(2)*[1;1], 1/sqrt(2)*[1;-1], 1/sqrt(2)*[1;-1im], 1/sqrt(2)*[1;1im] ] )
    
    exps = Matrix[ choi_liou_involution(vec(o)*vec(p)') for o in obs, p in prep ]

    pred = reduce(vcat, map(m->vec(m)', vec(exps)) )
    
    if trace(ρ) == 0.0
        ρ = choi_liou_involution(liou(rand_unitary(2))) 
    end
    
    asymptotic_means = real(pred*vec(ρ))

    samples = [ rand(Bernoulli((μ+1)/2),n) for μ in asymptotic_means ]
    sample_mean = 2*map(mean,samples)-1
    sample_var  = 4*map(var,samples)/n
    
    ρest, obj, status = qpt_ml(pred, sample_mean, sample_var);
    
    # get the normalization right
    choi_err = ρ-ρest |> choi_liou_involution |> liou2choi

    println("Status                  : $(status)")
    println("Diamond norm lower bound: $(snorm(choi_err,1))")
    println("χ² error                : $(obj)")
    println("Eigvals ρ:")
    for ev in eigvals(ρ)
        println(ev)
    end
    println("Eigvals ρest:")
    for ev in eigvals(ρest)
        println(real(ev))
    end
    
    return status, snorm(choi_err,1), obj, abs(choi_err), eigvals(ρ), eigvals(ρest)
end

function test_trb(da,db)
    r = rand_mixed_state(da*db)
    norm(trace(r,[da,db],2)-mat(trb_sop(da,db)*vec(r)),1)
end


ρ, obs = qst_test_setup()

kmax = 100
result = zeros(kmax,6)

srand(314159)

for k = 1:kmax

    ρ = .98*projector(rand(FubiniStudyPureState(2)))+.02*rand(HilbertSchmidtMixedState(2))
    
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
    
    #println(result[k,:])
end
@printf "\n"

#println(result)
