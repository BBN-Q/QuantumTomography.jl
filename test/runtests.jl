using Base.Test,
      Cliffords, 
      Distributions,
      QuantumInfo, 
      QuantumTomography, 
      RandomQuantum,
      SchattenNorms,
      SCS

#include("../src/QuantumTomography.jl")

function test_qst_ls_ideal()
    obs = Matrix[ (complex(Pauli(i))+eye(2))/2 for i in 1:3 ]
    append!(obs, Matrix[ (-complex(Pauli(i))+eye(2))/2 for i in 1:3 ])

    tomo = LSStateTomo(obs)

    ρ = .98*projector(rand(FubiniStudyPureState(2)))+.02*rand(HilbertSchmidtMixedState(2))
    
    ideal_means = real(predict(tomo,ρ))

    samples = ideal_means
    sample_mean = samples
    
    ρest, obj, status = fit(tomo, sample_mean, ones(length(samples)));
    
    return status, trnorm(ρ-ρest), obj, ρ, ρest
end

function test_qst_ls(n=10_000)
    
    obs = Matrix[ (complex(Pauli(i))+eye(2))/2 for i in 1:3 ]
    append!(obs, Matrix[ (-complex(Pauli(i))+eye(2))/2 for i in 1:3 ])

    tomo = LSStateTomo(obs)

    ρ = .98*projector(rand(FubiniStudyPureState(2)))+.02*rand(HilbertSchmidtMixedState(2))
    
    ideal_means = real(predict(tomo,ρ))
    
    samples = [ rand(Binomial(n,μ))/n for μ in ideal_means ]
    sample_mean = samples
    sample_var  = n*(samples - samples.^2)/(n-1)

    ρest, obj, status = fit(tomo, sample_mean, sample_var);
    
    return status, trnorm(ρ-ρest), obj, ρ, ρest
end

function test_qst_ml(n=10_000)
    obs = Matrix[ (complex(Pauli(i))+eye(2))/2 for i in 1:3 ]
    append!(obs, Matrix[ (-complex(Pauli(i))+eye(2))/2 for i in 1:3 ])
    #obs = map(real,Matrix[ (eye(2)+complex(Pauli(1)))/2, (eye(2)+complex(Pauli(2)))/2 ])
    #append!(obs, map(real,Matrix[ (eye(2)-complex(Pauli(1)))/2, (eye(2)-complex(Pauli(2)))/2 ]))

    tomo = MLStateTomo(obs)

    ρ = .99*[1. 1.; 1. 1.]/2+.01*[1 0; 0 0]#.98*projector(rand(FubiniStudyPureState(2)))+.02*rand(HilbertSchmidtMixedState(2))
    
    ideal_means = real(predict(tomo,ρ))

    samples = [rand(Binomial(n,μ)) for μ in ideal_means[1:3]]
    append!(samples,n-samples)

    ρest, obj, status = fit(tomo, samples, solver = SCSSolver(verbose=2))

    println(trnorm(ρ-ρest))
    
    return status, trnorm(ρ-ρest), obj, ρ, ρest
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

status, enorm, obj, ρ, ρest = test_qst_ls_ideal()
println("Constrained LS with ideal obs:")
println("   status: $status")
println("   error:  $enorm")
@test status == :Optimal
@test enorm < 1e-6

status, enorm, _, _, _ = test_qst_ls()
println("Constrained LS with realistic obs:")
println("   status: $status")
println("   error:  $enorm")
@test status == :Optimal
@test enorm < 1e-1

status, enorm, obj, ρ, ρest = test_qst_ml()
println(ρ)
println(ρest)
println("Strict ML with real obs:")
println("   status: $status")
println("   error:  $enorm")
@test status == :Optimal
@test enorm < 1e-2
