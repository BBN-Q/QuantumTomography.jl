using Base.Test,
      Cliffords, 
      ConicNonlinearBridge,
      Distributions,
      ECOS,
      Ipopt,
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

function test_qst_freels(ρ, obs; n=10_000, ideal=false)
    #obs = Matrix[ (complex(Pauli(i))+eye(2))/2 for i in 1:3 ]
    #append!(obs, Matrix[ (-complex(Pauli(i))+eye(2))/2 for i in 1:3 ])

    tomo = FreeLSStateTomo(obs)

    #ρ = .98*projector(rand(FubiniStudyPureState(2)))+.02*rand(HilbertSchmidtMixedState(2))
    
    ideal_means = real(predict(tomo,ρ))

    samples = ideal ? ideal_means : [ rand(Binomial(n,μ))/n for μ in ideal_means ]
    sample_mean = samples
    sample_var  = n*(samples - samples.^2)/(n-1)
    
    ρest, obj, status = fit(tomo, sample_mean, ideal ? ones(length(samples)) : sample_var);
    
    return status, trnorm(ρ-ρest), obj, ρest
end

function test_qst_ls(ρ, obs; n=10_000, ideal=false)
    #obs = Matrix[ (complex(Pauli(i))+eye(2))/2 for i in 1:3 ]
    #append!(obs, Matrix[ (-complex(Pauli(i))+eye(2))/2 for i in 1:3 ])

    tomo = LSStateTomo(obs)

    #ρ = .98*projector(rand(FubiniStudyPureState(2)))+.02*rand(HilbertSchmidtMixedState(2))
    
    ideal_means = real(predict(tomo,ρ))

    samples = ideal ? ideal_means : [ rand(Binomial(n,μ))/n for μ in ideal_means ]
    sample_mean = samples
    sample_var  = n*(samples - samples.^2)/(n-1)
    
    ρest, obj, status = fit(tomo, sample_mean, ideal ? ones(length(samples)) : sample_var);
    
    return status, trnorm(ρ-ρest), obj, ρest
end

function test_qst_ml(ρ, obs; n=10_000, β=0.0, ideal=false, alt=false, maxiter=100,ϵ=20)
    #obs = Matrix[ (complex(Pauli(i))+eye(2))/2 for i in 1:3 ]
    #append!(obs, Matrix[ (-complex(Pauli(i))+eye(2))/2 for i in 1:3 ])

    tomo = MLStateTomo(obs,β)

    #ρ = .98*projector(rand(FubiniStudyPureState(2)))+.02*rand(HilbertSchmidtMixedState(2))
    
    ideal_means = real(predict(tomo,ρ))
    
    samples = ideal ? ideal_means[1:3] : Float64[rand(Binomial(n,μ))/n for μ in ideal_means[1:3]]
    append!(samples,1-samples)

    ρest, obj, status = fit(tomo, samples, maxiter=maxiter, ϵ=ϵ)
    ρestB, objB, statusB = fitB(tomo, samples)

    println(abs(ρest-ρestB))

    #ρest, obj, status = alt ? fitB(tomo, samples) :
    #                          fitA(tomo, samples)

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
    
    ideal_means = real(pred*vec(ρ))

    samples = [ rand(Bernoulli((μ+1)/2),n) for μ in ideal_means ]
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

result = zeros(100,10)

for k = 1:100
ρ = .98*projector(rand(FubiniStudyPureState(2)))+.02*rand(HilbertSchmidtMixedState(2))

status, enorm, obj, ρest = test_qst_ls(ρ, obs, ideal=true)
result[k,1] = enorm
# println("Constrained LS with ∞ counts:")
# println("   status    : $status")
# println("   error     : $enorm")
# println("   true state: $ρ")
# println("   estimate  : $ρest")
# @test status == :Optimal
# @test enorm < 1e-6
# 
status, enorm, _, ρest = test_qst_ls(ρ, obs, ideal=false)
result[k,2] = enorm
# println("Constrained LS with 10_000 counts:")
# println("   status    : $status")
# println("   error     : $enorm")
# println("   true state: $ρ")
# println("   estimate  : $ρest")
# @test status == :Optimal
# @test enorm < 1e-1
# 
status, enorm, _, ρest = test_qst_ml(ρ, obs, ideal=true)
result[k,3] = enorm
# println("Strict ML with mean counts:")
# println("   status    : $status")
# println("   error     : $enorm")
# println("   true state: $ρ")
# println("   estimate  : $ρest")
# @test status == :Optimal
# @test enorm < 1e-2
# 
status, enorm, _, ρest = test_qst_ml(ρ, obs, ideal=true, alt=true)
result[k,4] = enorm
# println("Strict ML with mean counts:")
# println("   status    : $status")
# println("   error     : $enorm")
# println("   true state: $ρ")
# println("   estimate  : $ρest")
# @test status == :Optimal
# @test enorm < 1e-2
# 
status, enorm, _, ρest = test_qst_ml(ρ, obs, ideal=false)
result[k,5] = enorm
# println("Strict ML with 10_000 counts:")
# println("   status    : $status")
# println("   error     : $enorm")
# println("   true state: $ρ")
# println("   estimate  : $ρest")
# @test status == :Optimal
# @test enorm < 2e-2
# 
status, enorm, _, ρest = test_qst_ml(ρ, obs, ideal=false, alt=true)
result[k,6] = enorm
# println("Strict ML with 10_000 counts:")
# println("   status    : $status")
# println("   error     : $enorm")
# println("   true state: $ρ")
# println("   estimate  : $ρest")
# @test status == :Optimal
# @test enorm < 2e-2
# 
status, enorm, _, ρest = test_qst_ml(ρ, obs, β=0.04, ideal=true)
result[k,7] = enorm
# println("Hedged ML with mean counts and β=0.04:")
# println("   status    : $status")
# println("   error     : $enorm")
# println("   true state: $ρ")
# println("   estimate  : $ρest")
# @test status == :Optimal
# @test enorm < 2e-2
# 
status, enorm, _, ρest = test_qst_ml(ρ, obs, β=0.04, ideal=true, alt=true)
result[k,8] = enorm
# println("Hedged ML with mean counts and β=0.04:")
# println("   status    : $status")
# println("   error     : $enorm")
# println("   true state: $ρ")
# println("   estimate  : $ρest")
# @test status == :Optimal
# @test enorm < 2e-2
# 
status, enorm, _, ρest = test_qst_ml(ρ, obs, β=0.04, ideal=false)
result[k,9] = enorm
# println("Hedged ML with 10_000 counts and β=0.04:")
# println("   status    : $status")
# println("   error     : $enorm")
# println("   true state: $ρ")
# println("   estimate  : $ρest")
# @test status == :Optimal
# @test enorm < 2e-2
status, enorm, _, ρest = test_qst_ml(ρ, obs, β=0.04, ideal=false, alt=true)
result[k,10] = enorm
# println("Hedged ML with 10_000 counts and β=0.04:")
# println("   status    : $status")
# println("   error     : $enorm")
# println("   true state: $ρ")
# println("   estimate  : $ρest")
# @test status == :Optimal
# @test enorm < 2e-2
println(result[k,:])
end

println(result)
